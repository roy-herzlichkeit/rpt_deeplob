#!/usr/bin/env python3
"""Single-GPU step time vs batch size for DeepLOB (same model, data path and settings as train.py).

For each batch size: fresh model (same seed), Adam, 20 warm-up steps, then 200 measured steps.
The step loop is timed exactly like train.py: CUDA events around each phase, no sync inside a step
(so CPU launches overlap GPU work, as in real training):
  gather -> fwd (forward + loss) -> bwd (zero_grad + backward) -> opt (optimizer.step)
torch.cuda.synchronize() is called before and after the 200 measured steps.
step_ms = gather + fwd + bwd + opt per step (= t_step_ms in train.py's steps.csv); median and p95.
fwd_ms / bwd_ms / opt_ms = medians over the measured steps.
samples_per_s = batch * steps / wall time of the synchronized measured block.
peak_mem_mb = torch.cuda.max_memory_allocated over the measured steps (MB = 1e6 bytes);
  it includes the resident train data on the GPU (about 51 MB) and the model/Adam state.
A batch size that runs out of GPU memory gets "OOM" in its row and the sweep continues.
Rows are written to the CSV as they finish.

Run: python scripts/bench_batch.py        -> results/bench_<hostname>.csv
"""
import argparse
import csv
import socket
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as tr  # noqa: E402  (model, data loading, constants)

BATCHES = [16, 24, 32, 40, 48, 64, 96, 128, 192, 256, 384, 512]
COLUMNS = ["node", "batch", "step_ms_median", "step_ms_p95", "fwd_ms", "bwd_ms", "opt_ms",
           "samples_per_s", "peak_mem_mb"]
OOM_MARKERS = ("out of memory", "CUDNN_STATUS_ALLOC_FAILED", "CUBLAS_STATUS_ALLOC_FAILED")


def p95(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def make_run(train, batch, args, dev):
    torch.manual_seed(args.seed)
    model = tr.DeepLOB(3, head=args.head).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator()
    g.manual_seed(args.seed)
    n_steps = args.warmup + args.steps
    reps = -(-n_steps * batch // len(train))            # enough permutations to cover all steps
    perm = torch.cat([torch.randperm(len(train), generator=g) for _ in range(reps)]).to(dev)

    def step(i, ev=None):
        if ev:
            ev[0].record()
        xb, yb = train.batch(perm[i * batch:(i + 1) * batch], args.k_index)
        if ev:
            ev[1].record()
        loss = F.cross_entropy(model(xb), yb)
        if ev:
            ev[2].record()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if ev:
            ev[3].record()
        opt.step()
        if ev:
            ev[4].record()
    return step


def bench_one(train, batch, args, dev):
    step = make_run(train, batch, args, dev)
    for i in range(args.warmup):
        step(i)
    evs = [[torch.cuda.Event(enable_timing=True) for _ in range(5)] for _ in range(args.steps)]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for s in range(args.steps):
        step(args.warmup + s, evs[s])
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1e6
    st = [e[0].elapsed_time(e[4]) for e in evs]
    return {
        "step_ms_median": round(statistics.median(st), 3),
        "step_ms_p95": round(p95(st), 3),
        "fwd_ms": round(statistics.median(e[1].elapsed_time(e[2]) for e in evs), 3),
        "bwd_ms": round(statistics.median(e[2].elapsed_time(e[3]) for e in evs), 3),
        "opt_ms": round(statistics.median(e[3].elapsed_time(e[4]) for e in evs), 3),
        "samples_per_s": round(batch * args.steps / wall, 1),
        "peak_mem_mb": round(peak, 1),
    }


def is_oom(exc):
    return isinstance(exc, torch.OutOfMemoryError) or any(m in str(exc) for m in OOM_MARKERS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gpu-warmup-s", type=float, default=3.0,
                    help="seconds of discarded steps before the sweep (GPU clocks up)")
    ap.add_argument("--k-index", type=int, default=3, choices=range(5))
    ap.add_argument("--head", choices=["softmax", "logits"], default="logits")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--node", default=socket.gethostname(), help="value of the 'node' column")
    ap.add_argument("--out", default=None, help="default: results/bench_<hostname>.csv")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False             # same settings as train.py
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # train split exactly as in train.py (first 80% of the train file), kept on the GPU
    x, y = tr.load_file(tr.TRAIN_FILE)
    n_tr = int(x.shape[0] * (1 - tr.VAL_FRAC))
    train = tr.Split(torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), tr.windows_in(0, n_tr))

    out = Path(args.out) if args.out else tr.REPO / "results" / f"bench_{socket.gethostname()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"node {args.node} | {socket.gethostname()} | {torch.cuda.get_device_name(0)} | "
          f"torch {torch.__version__} | warmup {args.warmup} | steps {args.steps} | "
          f"k-index {args.k_index} | head {args.head}", flush=True)

    if args.gpu_warmup_s > 0:                          # discarded: lets the GPU reach its clocks
        step = make_run(train, 64, args, dev)
        t_end, i = time.perf_counter() + args.gpu_warmup_s, 0
        while time.perf_counter() < t_end:
            step(i % 3000)
            i += 1
            if i % 20 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        del step
        torch.cuda.empty_cache()

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for batch in [int(b) for b in args.batches.split(",")]:
            try:
                r = bench_one(train, batch, args, dev)
            except RuntimeError as exc:                # torch.OutOfMemoryError is a RuntimeError
                if not is_oom(exc):
                    raise
                r = {c: "OOM" for c in COLUMNS[2:]}
            torch.cuda.empty_cache()
            row = {"node": args.node, "batch": batch, **r}
            w.writerow(row)
            f.flush()
            print("  " + "  ".join(f"{c}={row[c]}" for c in COLUMNS), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
