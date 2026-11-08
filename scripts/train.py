#!/usr/bin/env python3
"""DeepLOB on FI-2010: single-GPU or 2-node DDP training with equal or balanced local batches.

single : python scripts/train.py --mode single --run-id RUN
2-node : torchrun --nnodes=2 --nproc_per_node=1 --node_rank=R --master_addr=192.168.10.1 \
           --master_port=29500 scripts/train.py --mode equal|balanced --backend nccl --run-id RUN

Data (FI-2010, NoAuction, DecPre): rows = 40 features + ... + 5 label rows, columns = time steps.
The whole dataset lives on the GPU; each batch is an index gather (no DataLoader).
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
CACHE_DIR = DATA_DIR / "cache"
T = 100                      # window length (time steps)
N_FEAT = 40                  # first 40 rows = 10-level LOB (price/volume, ask/bid)
N_LABEL_ROWS = 5             # last 5 rows = labels for 5 horizons
HORIZONS = [10, 20, 30, 50, 100]          # k-index 0..4 (ascending, see DeepLOB issue #30)
CLASS_NAMES = ["up", "stationary", "down"]  # FI-2010 labels 1/2/3, minus 1
TRAIN_FILE = "Train_Dst_NoAuction_DecPre_CF_7"
TEST_FILES = ["Test_Dst_NoAuction_DecPre_CF_7",
              "Test_Dst_NoAuction_DecPre_CF_8",
              "Test_Dst_NoAuction_DecPre_CF_9"]
VAL_FRAC = 0.2
GAP = 100                    # unused time steps between the train and val parts
RESUME_MUST_MATCH = ["mode", "global_batch", "local_batches", "lr", "seed", "k_index", "head",
                     "max_steps"]


# ----------------------------------------------------------------------------- data
def load_file(name):
    """Return X [N, 40] float32 and Y [N, 5] int64 (labels 0..2). Cached as .npy in data/cache/."""
    fx, fy = CACHE_DIR / f"{name}_X.npy", CACHE_DIR / f"{name}_Y.npy"
    if fx.exists() and fy.exists():
        x, y = np.load(fx), np.load(fy)
        ok = (x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == N_FEAT
              and y.dtype == np.int64 and y.shape == (x.shape[0], N_LABEL_ROWS)
              and set(np.unique(y).tolist()) <= {0, 1, 2})
        if ok:
            return x, y
        print(f"cache for {name} is invalid; re-parsing the .txt", flush=True)
    import pandas as pd
    src = DATA_DIR / f"{name}.txt"
    raw = pd.read_csv(src, sep=r"\s+", header=None, dtype=np.float64).to_numpy()
    x = np.ascontiguousarray(raw[:N_FEAT].T, dtype=np.float32)
    lab = raw[-N_LABEL_ROWS:].T
    if not set(np.unique(lab).tolist()) <= {1.0, 2.0, 3.0}:
        raise ValueError(f"{src}: unexpected label values {np.unique(lab)}")
    y = np.ascontiguousarray(lab - 1, dtype=np.int64)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path, arr in ((fx, x), (fy, y)):
        tmp = path.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)
    return x, y


class Split:
    """One split on the GPU. A sample is identified by the row index of its LAST time step."""

    def __init__(self, x, y, ends):
        self.x, self.y = x, y                                   # GPU tensors [N,40], [N,5]
        self.ends = torch.as_tensor(ends, dtype=torch.long, device=x.device)
        self.offs = torch.arange(-(T - 1), 1, device=x.device)  # [T]: -99..0

    def __len__(self):
        return self.ends.numel()

    def batch(self, sample_idx, k):
        e = self.ends[sample_idx]                      # [B]
        rows = e[:, None] + self.offs[None, :]         # [B, T]
        return self.x[rows].unsqueeze(1), self.y[e, k]  # [B,1,T,40], [B]


def windows_in(start, stop):
    """End rows of all T-windows that lie fully inside rows [start, stop)."""
    return np.arange(start + T - 1, stop)


def build_data(device):
    xtr, ytr = load_file(TRAIN_FILE)
    n = xtr.shape[0]
    n_tr = int(np.floor(n * (1 - VAL_FRAC)))           # same cut as the notebook
    gx, gy = torch.from_numpy(xtr).to(device), torch.from_numpy(ytr).to(device)
    train = Split(gx, gy, windows_in(0, n_tr))
    val = Split(gx, gy, windows_in(n_tr + GAP, n))

    xs, ys, ends, off, per_file = [], [], [], 0, []
    for name in TEST_FILES:                            # windows built inside each file
        x, y = load_file(name)
        xs.append(x)
        ys.append(y)
        ends.append(windows_in(off, off + x.shape[0]))
        per_file.append({"file": name, "rows": int(x.shape[0]), "windows": int(len(ends[-1]))})
        off += x.shape[0]
    test = Split(torch.from_numpy(np.concatenate(xs)).to(device),
                 torch.from_numpy(np.concatenate(ys)).to(device), np.concatenate(ends))

    info = {
        "train_file_rows": int(n), "train_rows": [0, n_tr], "val_rows": [n_tr + GAP, n],
        "gap_rows": GAP, "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "test_files": per_file,
        # sha256 of the parsed arrays, compared across ranks (each node has its own data/)
        "fingerprint": {name: sha256_arrays(x, y) for name, x, y in
                        [(TRAIN_FILE, xtr, ytr)] + list(zip(TEST_FILES, xs, ys))},
    }
    return train, val, test, info


def sha256_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(str((a.dtype, a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def class_counts(split, k):
    c = torch.bincount(split.y[split.ends, k], minlength=3)
    return c.tolist()


# ----------------------------------------------------------------------------- model
class DeepLOB(nn.Module):
    """DeepLOB as in the authors' PyTorch notebook (run_train_pytorch.ipynb).

    head='logits' (default) returns raw logits for nn.CrossEntropyLoss. head='softmax' returns
    softmax probabilities, exactly like the notebook (which then applies CrossEntropyLoss on top).
    """

    def __init__(self, y_len=3, head="logits"):
        super().__init__()
        self.head = head
        lrelu = lambda: nn.LeakyReLU(negative_slope=0.01)  # noqa: E731
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)), lrelu(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), lrelu(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), lrelu(), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)), lrelu(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), lrelu(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)), lrelu(), nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), lrelu(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"), lrelu(), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), lrelu(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"), lrelu(), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), lrelu(), nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, y_len)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64, device=x.device)
        c0 = torch.zeros(1, x.size(0), 64, device=x.device)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.cat((self.inp1(x), self.inp2(x), self.inp3(x)), dim=1)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc1(x[:, -1, :])
        return torch.softmax(x, dim=1) if self.head == "softmax" else x


# ----------------------------------------------------------------------------- helpers
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "base.yaml"),
                    help="YAML with defaults for global_batch, epochs, seed, lr")
    ap.add_argument("--mode", choices=["single", "equal", "balanced"], required=True)
    ap.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    ap.add_argument("--global-batch", type=int, default=None)
    ap.add_argument("--local-batches", default=None,
                    help="comma list in rank order, e.g. 40,24 (balanced). equal = G/world each")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap on optimizer steps per epoch (smoke tests)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<run_id>/last.pt")
    ap.add_argument("--k-index", type=int, default=3, choices=range(5),
                    help="label row 0..4 = horizon k=10,20,30,50,100 (default 3 = k50)")
    ap.add_argument("--head", choices=["softmax", "logits"], default="logits",
                    help="logits = plain CE (default); softmax = notebook (CE on softmax output)")
    ap.add_argument("--cudnn-benchmark", choices=["on", "off"], default="on",
                    help="cuDNN autotuning (on: autotune happens in the untimed warm-up)")
    ap.add_argument("--eval-batch", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if args.config and Path(args.config).exists() else {}
    for key, cfg_key in (("global_batch", "global_batch"), ("epochs", "epochs"),
                         ("seed", "seed"), ("lr", "lr")):
        if getattr(args, key) is None:
            val = cfg.get(cfg_key)
            if val is None or val == "TODO":
                ap.error(f"--{key.replace('_', '-')} not given and not set in {args.config}")
            setattr(args, key, float(val) if key == "lr" else int(val))
    if args.mode == "balanced" and args.local_batches is None and cfg.get("local_batches_balanced"):
        args.local_batches = str(cfg["local_batches_balanced"])
    return args


def resolve_local_batches(args, world):
    G = args.global_batch
    if args.mode == "single":
        if world != 1:
            raise SystemExit("--mode single must run as one process (plain python, no torchrun)")
        return [G]
    if world < 2:
        raise SystemExit(f"--mode {args.mode} needs torchrun with >= 2 processes")
    if args.mode == "equal":
        if G % world:
            raise SystemExit(f"--global-batch {G} is not divisible by world size {world}")
        lb = [G // world] * world
        if args.local_batches and [int(v) for v in args.local_batches.split(",")] != lb:
            raise SystemExit(f"--mode equal uses {lb}; remove --local-batches or use --mode balanced")
        return lb
    if not args.local_batches:
        raise SystemExit("--mode balanced needs --local-batches, e.g. 40,24")
    lb = [int(v) for v in args.local_batches.split(",")]
    if len(lb) != world or sum(lb) != G or min(lb) <= 0:
        raise SystemExit(f"--local-batches {lb}: need {world} positive values that sum to {G}")
    return lb


def git_info():
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    # dirty = tracked changes anywhere, or untracked files in the code folders
    return {"commit": run("git", "rev-parse", "HEAD") or "unknown",
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no")
                          or run("git", "status", "--porcelain", "--", "scripts", "src", "configs"))}


def link_speed():
    """Speed of the interface used for DDP traffic."""
    spec = os.environ.get("NCCL_SOCKET_IFNAME") or os.environ.get("GLOO_SOCKET_IFNAME") or ""
    iface = spec.split(",")[0].lstrip("=")
    if not iface or iface.startswith("^") or not Path(f"/sys/class/net/{iface}").exists():
        # exclusion list / prefix / unset: use the interface that routes to the master
        iface = "eno1"
        master = os.environ.get("MASTER_ADDR")
        if master:
            try:
                out = subprocess.run(["ip", "-o", "route", "get", master], capture_output=True,
                                     text=True, timeout=5).stdout.split()
                if "dev" in out:
                    iface = out[out.index("dev") + 1]
            except Exception:  # noqa: BLE001
                pass
    try:
        speed = Path(f"/sys/class/net/{iface}/speed").read_text().strip() + " Mb/s"
    except OSError:
        speed = "unknown"
    return {"iface": iface, "speed": speed}


def rank_info(rank):
    p = torch.cuda.get_device_properties(0)
    return {"rank": rank, "hostname": socket.gethostname(), "gpu": p.name,
            "gpu_memory_gb": round(p.total_memory / 1024 ** 3, 2), "link": link_speed(),
            "git": git_info(), "torch": torch.__version__, "cuda": torch.version.cuda}


def gather(obj, distributed, world):
    if not distributed:
        return [obj]
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(0),
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state(s["cuda"], 0)
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def truncate_csv(path, max_epoch):
    """On resume: drop rows of epochs that were not checkpointed."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    kept = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= max_epoch]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(kept)


def append_csv(path, header, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


@torch.no_grad()
def evaluate(model, split, k, head, bs):
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), dtype=torch.float64, device=split.x.device)
    preds = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.x.device)
        xb, yb = split.batch(idx, k)
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").double()
        preds.append(out.argmax(1))
    model.train(was_training)
    y_true = split.y[split.ends, k].cpu().numpy()
    y_pred = torch.cat(preds).cpu().numpy()
    return {
        "loss": loss_sum.item() / len(split),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "f1_per_class": [float(v) for v in f1_score(y_true, y_pred, average=None, labels=[0, 1, 2],
                                                     zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "n": int(len(split)),
    }


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = args.mode != "single"
    local_batches = resolve_local_batches(args, world)
    G = args.global_batch
    lb = local_batches[rank]
    lo = sum(local_batches[:rank])
    hi = lo + lb

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == "on"
    torch.backends.cudnn.allow_tf32 = False          # strict fp32 on both GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    if distributed:
        extra = {"device_id": dev} if args.backend == "nccl" else {}
        dist.init_process_group(backend=args.backend, timeout=datetime.timedelta(minutes=30), **extra)
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # ---- startup checks, shared by all ranks so that no rank exits alone and leaves the other
    # one waiting in a collective. Every rank must run with the same settings and code.
    run_id = args.run_id or f"{args.mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    err = None
    if is_main:
        ld, cd = REPO / "logs" / run_id, REPO / "checkpoints" / run_id
        if args.resume and not (cd / "last.pt").exists():
            err = f"--resume: {cd / 'last.pt'} not found"
        elif not args.resume and (ld.exists() or cd.exists()):
            err = f"run '{run_id}' already exists; use a new --run-id or --resume"
    agree = {key: vars(args)[key] for key in ("mode", "backend", "global_batch", "epochs", "lr",
                                             "seed", "max_steps", "k_index", "head", "resume",
                                             "run_id", "eval_batch", "cudnn_benchmark")}
    agree.update(local_batches=local_batches, world_size=world, torch=torch.__version__,
                 commit=git_info()["commit"])
    checks = gather({"err": err, "agree": agree, "run_id": run_id}, distributed, world)
    errors = [c["err"] for c in checks if c["err"]]
    differ = {key: [c["agree"][key] for c in checks] for key in agree
              if any(c["agree"][key] != checks[0]["agree"][key] for c in checks)}
    if differ:
        errors.append(f"ranks disagree (values in rank order): {differ}")
    if errors:
        raise SystemExit("startup check failed: " + " | ".join(errors))
    run_id = checks[0]["run_id"]                      # rank 0's id (matters if --run-id omitted)
    log_dir, ckpt_dir = REPO / "logs" / run_id, REPO / "checkpoints" / run_id
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- data
    train, val, test, data_info = build_data(dev)
    k = args.k_index
    steps_per_epoch = len(train) // G
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)

    # ---- model
    model = DeepLOB(3, head=args.head).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- resume (rank 0 reads last.pt and sends it to the other ranks)
    start_epoch, best_val, best_epoch = 1, float("inf"), 0
    if args.resume:
        ckpt = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False) if is_main else None
        if distributed:
            box = [ckpt]
            dist.broadcast_object_list(box, src=0)
            ckpt = box[0]
        for key in RESUME_MUST_MATCH:
            want = local_batches if key == "local_batches" else getattr(args, key)
            have = ckpt["derived"]["local_batches"] if key == "local_batches" else ckpt["args"][key]
            if want != have:
                raise SystemExit(f"--resume: {key}={want} but checkpoint has {have}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng"])
        start_epoch = ckpt["epoch"] + 1
        best_val, best_epoch = ckpt["best_val_loss"], ckpt["best_epoch"]
        if is_main:
            truncate_csv(log_dir / "steps.csv", ckpt["epoch"])
            truncate_csv(log_dir / "epochs.csv", ckpt["epoch"])
        log(f"resumed from {ckpt_dir / 'last.pt'}: last completed epoch {ckpt['epoch']}, "
            f"continuing at epoch {start_epoch}")

    # ---- warm-up: build cuDNN/cuBLAS kernels on a throwaway copy (real weights, BN stats, RNG
    # untouched) so the first timed step does not include one-time CUDA initialisation
    warm = copy.deepcopy(model).train()
    warm.lstm.flatten_parameters()       # deepcopy breaks cuDNN's contiguous LSTM weight buffer
    wx, wy = train.batch(torch.arange(lb, device=dev), k)
    for _ in range(3):
        F.cross_entropy(warm(wx), wy).backward()
    with torch.no_grad():
        warm.eval()(val.batch(torch.arange(min(args.eval_batch, len(val)), device=dev), k)[0])
    del warm, wx, wy
    torch.cuda.synchronize()

    # broadcast_buffers=False: no per-step BN-buffer broadcast. Rank 0's weights, running stats,
    # checkpoints and val/test results are identical either way (rank 0 is the broadcast source,
    # and train-mode BN uses batch statistics); it only removes one collective per step from t_fwd.
    net = DDP(model, device_ids=[0], broadcast_buffers=False) if distributed else model

    # ---- config.json (all ranks)
    ranks = gather(rank_info(rank), distributed, world)
    prints = gather(data_info["fingerprint"], distributed, world)
    if any(p != prints[0] for p in prints):
        raise SystemExit(f"data differs between ranks (sha256 per file): {prints}")
    notes = [
        "BatchNorm: local per-rank batch statistics (no SyncBatchNorm); DDP broadcast_buffers=False, "
        "so each rank keeps its own running stats; rank 0's are used for val, test and checkpoints",
        f"fp32 only: no AMP, TF32 disabled for matmul and cuDNN; cudnn.benchmark={args.cudnn_benchmark}",
        "2-node loss: CE(sum over local samples) / G * world_size; DDP averages over world_size "
        "-> exact mean gradient of the global batch",
        f"head={args.head}: " + ("softmax output + CrossEntropyLoss, as in the notebook"
                                 if args.head == "softmax" else "logits + CrossEntropyLoss"),
        "steps.csv: t_fwd = forward+loss, t_bwd = backward (incl. DDP all_reduce), t_opt = "
        "optimizer.step, t_step = batch gather + fwd + bwd + opt (CUDA events); loss = local mean CE",
        "epochs.csv: time_s = max over ranks of the training-loop wall time (no eval)",
        "--max-steps caps steps per epoch",
        "warm-up: 3 fwd/bwd + 1 eval forward on a deep copy of the model before training, with the "
        "same shapes as training (cuDNN autotune happens here, not in timed steps) "
        "(kernel init only; weights, BN stats and RNG unchanged)",
    ]
    config = {
        "run_id": run_id, "args": vars(args), "argv": sys.argv, "world_size": world,
        "local_batches": local_batches, "steps_per_epoch": steps_per_epoch,
        "horizon_k": HORIZONS[k], "class_names": CLASS_NAMES,
        "param_count": n_params, "grad_bytes": grad_bytes, "grad_mb": grad_bytes / 1e6,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(),
                     "nccl": ".".join(map(str, torch.cuda.nccl.version()))},
        "ranks": ranks, "data": data_info,
        "class_counts_selected_k": {s: class_counts(sp, k) for s, sp in
                                    (("train", train), ("val", val), ("test", test))},
        "class_counts_all_k": {f"k_index {j} (k={HORIZONS[j]})":
                               {s: class_counts(sp, j) for s, sp in
                                (("train", train), ("val", val), ("test", test))}
                               for j in range(N_LABEL_ROWS)},
        "env": {kk: v for kk, v in os.environ.items()
                if kk.startswith(("NCCL_", "GLOO_", "MASTER_", "TORCH_NCCL"))},
        "notes": notes,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if is_main:
        name = "config.json" if not args.resume else f"config_resume_e{start_epoch}.json"
        (log_dir / name).write_text(json.dumps(config, indent=2))
    log(f"run {run_id} | mode {args.mode} | world {world} | local batches {local_batches} | "
        f"G {G} | lr {args.lr} | k-index {k} (k={HORIZONS[k]}) | head {args.head}")
    log(f"hosts: {[(r['rank'], r['hostname'], r['gpu'], r['link']['speed']) for r in ranks]}")
    log(f"params {n_params:,} | gradient {grad_bytes / 1e6:.3f} MB ({grad_bytes / 2**20:.3f} MiB)")
    log(f"samples: train {len(train)} | val {len(val)} | test {len(test)} | "
        f"steps/epoch {steps_per_epoch}")
    log(notes[0])

    step_header = ["epoch", "step", "rank", "local_batch", "t_fwd_ms", "t_bwd_ms", "t_opt_ms",
                   "t_step_ms", "loss"]
    epoch_header = ["epoch", "time_s", "samples_per_s", "train_loss", "val_loss", "val_acc",
                    "val_f1_macro"]
    scale = world / G  # 2-node loss scaling

    for epoch in range(start_epoch, args.epochs + 1):
        net.train()
        g = torch.Generator()
        g.manual_seed(args.seed + epoch)
        perm_cpu = torch.randperm(len(train), generator=g)       # must be identical on every rank
        if distributed:
            digests = gather(sha256_arrays(perm_cpu.numpy()), distributed, world)
            if any(d != digests[0] for d in digests):
                raise SystemExit(f"epoch {epoch}: train permutation differs between ranks")
        perm = perm_cpu.to(dev)
        ev = [[torch.cuda.Event(enable_timing=True) for _ in range(5)] for _ in range(steps_per_epoch)]
        step_loss = torch.zeros(steps_per_epoch, device=dev)
        loss_sum = torch.zeros((), dtype=torch.float64, device=dev)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(steps_per_epoch):
            e = ev[s]
            e[0].record()
            idx = perm[s * G:(s + 1) * G][lo:hi]
            xb, yb = train.batch(idx, k)
            e[1].record()
            out = net(xb)
            ce_sum = F.cross_entropy(out, yb, reduction="sum")
            loss = ce_sum * scale if distributed else ce_sum / lb
            e[2].record()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            e[3].record()
            optimizer.step()
            e[4].record()
            step_loss[s] = ce_sum.detach() / lb
            loss_sum += ce_sum.detach().double()
            if is_main and args.log_every and (s + 1) % args.log_every == 0:
                print(f"  epoch {epoch} step {s + 1}/{steps_per_epoch} loss {step_loss[s].item():.4f}",
                      flush=True)
        torch.cuda.synchronize()
        train_time = time.perf_counter() - t0

        losses = step_loss.tolist()
        rows = []
        for s in range(steps_per_epoch):
            e = ev[s]
            rows.append([epoch, s + 1, rank, lb,
                         round(e[1].elapsed_time(e[2]), 4), round(e[2].elapsed_time(e[3]), 4),
                         round(e[3].elapsed_time(e[4]), 4), round(e[0].elapsed_time(e[4]), 4),
                         round(losses[s], 6)])
        per_rank = gather({"rows": rows, "loss_sum": loss_sum.item(), "n": steps_per_epoch * lb,
                           "time": train_time}, distributed, world)

        if is_main:
            append_csv(log_dir / "steps.csv", step_header,
                       [r for pr in per_rank for r in pr["rows"]])
            time_s = max(pr["time"] for pr in per_rank)
            train_loss = sum(pr["loss_sum"] for pr in per_rank) / sum(pr["n"] for pr in per_rank)
            vm = evaluate(model, val, k, args.head, args.eval_batch)
            append_csv(log_dir / "epochs.csv", epoch_header,
                       [[epoch, round(time_s, 3), round(steps_per_epoch * G / time_s, 2),
                         round(train_loss, 6), round(vm["loss"], 6), round(vm["acc"], 6),
                         round(vm["f1_macro"], 6)]])
            if not math.isfinite(vm["loss"]):
                print(f"WARNING: epoch {epoch} val loss is {vm['loss']}", flush=True)
            improved = vm["loss"] < best_val
            if improved:
                best_val, best_epoch = vm["loss"], epoch
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_val_loss": best_val, "best_epoch": best_epoch,
                    "rng": rng_state(), "args": vars(args),
                    "derived": {"local_batches": local_batches, "world_size": world,
                                "steps_per_epoch": steps_per_epoch}}
            # best.pt first: last.pt must never claim a best epoch that is not on disk.
            # (best.pt is also written if it does not exist yet, e.g. when val loss is NaN.)
            if improved or not (ckpt_dir / "best.pt").exists():
                save_atomic(ckpt, ckpt_dir / "best.pt")
            save_atomic(ckpt, ckpt_dir / "last.pt")
            print(f"epoch {epoch}/{args.epochs} | {time_s:.1f} s | "
                  f"{steps_per_epoch * G / time_s:.0f} samples/s | train_loss {train_loss:.4f} | "
                  f"val_loss {vm['loss']:.4f} | val_acc {vm['acc']:.4f} | "
                  f"val_f1 {vm['f1_macro']:.4f}{' | best' if improved else ''}", flush=True)
        if distributed:
            dist.barrier()

    # ---- test (rank 0, best.pt)
    if is_main and not (ckpt_dir / "best.pt").exists():
        print(f"no {ckpt_dir / 'best.pt'} (no epoch was trained); skipping the test", flush=True)
    elif is_main:
        best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
        test_model = DeepLOB(3, head=args.head).to(dev)
        test_model.load_state_dict(best["model"])
        tm = evaluate(test_model, test, k, args.head, args.eval_batch)
        result = {"run_id": run_id, "checkpoint": str(ckpt_dir / "best.pt"),
                  "best_epoch": best["epoch"], "best_val_loss": best["best_val_loss"],
                  "k_index": k, "horizon_k": HORIZONS[k], "class_names": CLASS_NAMES,
                  "accuracy": tm["acc"], "f1_macro": tm["f1_macro"],
                  "f1_per_class": dict(zip(CLASS_NAMES, tm["f1_per_class"])),
                  "confusion_matrix": tm["confusion_matrix"], "test_loss": tm["loss"],
                  "n_test": tm["n"]}
        (log_dir / "test.json").write_text(json.dumps(result, indent=2))
        print(f"test (best.pt, epoch {best['epoch']}): acc {tm['acc']:.4f} | "
              f"f1_macro {tm['f1_macro']:.4f} | n {tm['n']}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
