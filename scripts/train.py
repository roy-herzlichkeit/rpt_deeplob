#!/usr/bin/env python3
"""DeepLOB on FI-2010: single-GPU or 2-node DDP training with equal or balanced local batches.

single : python scripts/train.py --mode single --run-id RUN
2-node : torchrun --nnodes=2 --nproc_per_node=1 --node_rank=R --master_addr=192.168.10.1 \
           --master_port=29500 scripts/train.py --mode equal|balanced --backend nccl --run-id RUN

Data (FI-2010, NoAuction, DecPre): rows = 40 features + ... + 5 label rows, columns = time steps.
The whole dataset lives on the GPU; each batch is an index gather (no DataLoader).
"""
import argparse
import contextlib
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
PG_TIMEOUT_S = 120          # a rank that loses its peer must exit, not hang
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
    ap.add_argument("--resume", action="store_true",
                    help="continue at the saved epoch and step from checkpoints/<run_id>/rank<r>/last.pt\n"
                         "(last.pt.prev if last.pt is missing or unreadable)")
    ap.add_argument("--k-index", type=int, default=3, choices=range(5),
                    help="label row 0..4 = horizon k=10,20,30,50,100 (default 3 = k50)")
    ap.add_argument("--head", choices=["softmax", "logits"], default="logits",
                    help="logits = plain CE (default); softmax = notebook (CE on softmax output)")
    ap.add_argument("--cudnn-benchmark", choices=["on", "off"], default="on",
                    help="cuDNN autotuning (on: autotune happens in the untimed warm-up)")
    ap.add_argument("--no-sync", action="store_true",
                    help="TIMING ONLY: run every step inside DDP no_sync(), i.e. no gradient exchange;\n"
                         "ranks drift apart. No checkpoints, no validation, no test.")
    ap.add_argument("--ckpt-every", type=int, default=1000,
                    help="also save last.pt every N steps inside an epoch (0 = only at epoch end)")
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


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_durable(obj, path, keep_prev=False):
    """Crash-safe save: write <name>.tmp, flush + fsync, (optionally move the current file to
    <name>.prev), os.replace tmp -> path, then fsync the folder so the rename itself is on disk."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if keep_prev and path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)
    fsync_dir(path.parent)


CKPT_KEYS = ("model", "optimizer", "epoch", "step", "epoch_done", "rng", "args", "derived",
             "best_val_loss", "best_epoch", "progress")


def load_ckpt(path):
    """Return the checkpoint dict, or None if the file is missing, truncated or incomplete."""
    if not path.exists():
        return None
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001  (truncated / corrupt file)
        return None
    return c if isinstance(c, dict) and all(k in c for k in CKPT_KEYS) else None


def ckpt_pos(c):
    """Training position of a checkpoint: (epoch, steps done in that epoch, epoch finished)."""
    return (int(c["epoch"]), int(c["step"]), bool(c["epoch_done"]))


def state_digest(model, optimizer):
    """sha256 of all parameters and optimizer tensors (BatchNorm buffers excluded: they differ
    per rank by design)."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    for st in optimizer.state_dict()["state"].values():
        for key in sorted(st):
            v = st[key]
            h.update(v.detach().cpu().numpy().tobytes() if torch.is_tensor(v) else repr(v).encode())
    return h.hexdigest()


def filter_csv(path, keep):
    """On resume: keep the header and the rows for which keep(row) is true (durable rewrite)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if r and keep(r)])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv(path, header, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


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
        # NCCL: on a timed-out or failed collective, tear the process down (do not hang)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "3")
        extra = {"device_id": dev} if args.backend == "nccl" else {}
        dist.init_process_group(backend=args.backend,
                                timeout=datetime.timedelta(seconds=PG_TIMEOUT_S), **extra)
    is_main = rank == 0

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # ---- startup checks, shared by all ranks so that no rank exits alone and leaves the other
    # one waiting in a collective. Every rank must run with the same settings and code.
    # Each rank checks its OWN disk (every node keeps its own logs/ and checkpoints/).
    run_id = args.run_id or f"{args.mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    if distributed and not args.run_id:
        box = [run_id]
        dist.broadcast_object_list(box, src=0)          # rank 0's id on every rank
        run_id = box[0]
    log_dir, ckpt_dir = REPO / "logs" / run_id, REPO / "checkpoints" / run_id
    # this rank's own checkpoint folder (separate per rank, so it is also safe when two ranks
    # share a disk, e.g. a local test); best.pt (rank 0) stays in ckpt_dir
    rank_dir = ckpt_dir / f"rank{rank}"
    err = None
    if args.no_sync and not distributed:
        err = "--no-sync needs a 2-node mode (equal or balanced)"
    elif args.no_sync and args.resume:
        err = "--no-sync runs keep no checkpoints; start them again instead of --resume"
    elif args.resume and not any((rank_dir / n).exists() for n in ("last.pt", "last.pt.prev")):
        err = f"rank {rank}: --resume but no {rank_dir}/last.pt or last.pt.prev"
    elif not args.resume and (log_dir.exists() or ckpt_dir.exists()):
        err = f"rank {rank}: run '{run_id}' already exists; use a new --run-id or --resume"
    agree = {key: vars(args)[key] for key in ("mode", "backend", "global_batch", "epochs", "lr",
                                             "seed", "max_steps", "k_index", "head", "resume",
                                             "run_id", "eval_batch", "cudnn_benchmark",
                                             "ckpt_every", "no_sync")}
    agree.update(local_batches=local_batches, world_size=world, torch=torch.__version__,
                 commit=git_info()["commit"])
    checks = gather({"err": err, "agree": agree}, distributed, world)
    errors = [c["err"] for c in checks if c["err"]]
    differ = {key: [c["agree"][key] for c in checks] for key in agree
              if any(c["agree"][key] != checks[0]["agree"][key] for c in checks)}
    if differ:
        errors.append(f"ranks disagree (values in rank order): {differ}")
    if errors:
        raise SystemExit("startup check failed: " + " | ".join(errors))
    log_dir.mkdir(parents=True, exist_ok=True)
    rank_dir.mkdir(parents=True, exist_ok=True)
    steps_csv, epochs_csv = log_dir / f"steps_rank{rank}.csv", log_dir / f"epochs_rank{rank}.csv"

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
    derived = {"local_batches": local_batches, "world_size": world, "steps_per_epoch": steps_per_epoch}

    # ---- resume: every rank reads its own last.pt (or last.pt.prev if last.pt is missing/bad).
    # The ranks then agree on the newest position that ALL of them have, so a kill during a save
    # on one node cannot make the ranks restart from different steps.
    start_epoch, start_step, best_val, best_epoch = 1, 0, float("inf"), 0
    progress = {"loss_sum": 0.0, "n": 0, "time": 0.0, "ckpt_s": 0.0}
    if args.resume:
        cands = {}
        for name in ("last.pt.prev", "last.pt"):        # last.pt wins a tie
            c = load_ckpt(rank_dir / name)
            if c is not None:
                cands[ckpt_pos(c)] = (name, c)
            elif (rank_dir / name).exists():
                print(f"rank {rank}: {rank_dir / name} is unreadable or incomplete; ignoring it",
                      flush=True)
        offered = gather(sorted(cands), distributed, world)
        common = set(offered[0]).intersection(*map(set, offered[1:]))
        if not common:
            raise SystemExit(f"--resume: no checkpoint position common to all ranks: {offered}")
        pos = max(common)
        name, ckpt = cands[pos]
        for key in RESUME_MUST_MATCH:
            want = local_batches if key == "local_batches" else getattr(args, key)
            have = ckpt["derived"]["local_batches"] if key == "local_batches" else ckpt["args"][key]
            if want != have:
                raise SystemExit(f"--resume: {key}={want} but checkpoint has {have}")
        model.load_state_dict(ckpt["model"])            # this rank's own BatchNorm buffers
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng"])
        best_val, best_epoch = ckpt["best_val_loss"], ckpt["best_epoch"]
        ep, st, done = pos
        if done or st >= steps_per_epoch:
            start_epoch, start_step = ep + 1, 0
        else:
            start_epoch, start_step, progress = ep, st, dict(ckpt["progress"])
        digests = gather(state_digest(model, optimizer), distributed, world)
        if any(d != digests[0] for d in digests):
            raise SystemExit("--resume: weights/optimizer differ between ranks after loading")
        completed = ep if done else ep - 1
        filter_csv(steps_csv, lambda r: (int(r[0]), int(r[1])) <= (ep, st))
        filter_csv(epochs_csv, lambda r: int(r[0]) <= completed)
        print(f"rank {rank}: resumed from {rank_dir / name} (epoch {ep}, step {st}"
              f"{', epoch finished' if done else ''}) -> continuing at epoch {start_epoch}, "
              f"step {start_step + 1}", flush=True)

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

    # broadcast_buffers=False: no per-step BN-buffer broadcast, and DDP's constructor does not
    # overwrite the BatchNorm buffers either, so each rank keeps (and resumes) its own running
    # stats. Parameters are broadcast from rank 0 at construction (identical anyway, see digest).
    net = DDP(model, device_ids=[0], broadcast_buffers=False) if distributed else model

    # ---- config.json (every rank writes its own copy)
    ranks = gather(rank_info(rank), distributed, world)
    prints = gather(data_info["fingerprint"], distributed, world)
    if any(p != prints[0] for p in prints):
        raise SystemExit(f"data differs between ranks (sha256 per file): {prints}")
    notes = [
        "BatchNorm: local per-rank batch statistics (no SyncBatchNorm); DDP broadcast_buffers=False, "
        "so each rank keeps its own running stats (saved in its own checkpoint); rank 0's are used "
        "for val, test and best.pt",
        f"fp32 only: no AMP, TF32 disabled for matmul and cuDNN; cudnn.benchmark={args.cudnn_benchmark}",
        "2-node loss: CE(sum over local samples) / G * world_size; DDP averages over world_size "
        "-> exact mean gradient of the global batch",
        f"head={args.head}: " + ("softmax output + CrossEntropyLoss, as in the notebook"
                                 if args.head == "softmax" else "logits + CrossEntropyLoss"),
        "steps_rank<r>.csv: t_fwd = forward+loss, t_bwd = backward (incl. DDP all_reduce), t_opt = "
        "optimizer.step, t_step = batch gather + fwd + bwd + opt (CUDA events); loss = local mean "
        "CE; rows are written at every checkpoint and at the end of each epoch",
        "epochs_rank<r>.csv: time_s = max over ranks of the training-loop time without checkpoint "
        "saves and without downtime between a crash and --resume; ckpt_s = this rank's time spent "
        "saving mid-epoch checkpoints; val metrics come from rank 0",
        f"checkpoints: every rank saves checkpoints/<run_id>/rank<r>/last.pt every {args.ckpt_every} steps "
        "and at the end of every epoch (tmp + fsync + rename + folder fsync; previous kept as "
        "last.pt.prev); rank 0 also keeps best.pt",
        f"process group timeout {PG_TIMEOUT_S} s: a rank that loses its peer exits",
        "--max-steps caps steps per epoch",
        "warm-up: 3 fwd/bwd + 1 eval forward on a deep copy of the model before training, with the "
        "same shapes as training (cuDNN autotune happens here, not in timed steps) "
        "(kernel init only; weights, BN stats and RNG unchanged)",
    ]
    config = {
        "run_id": run_id, "rank": rank, "args": vars(args), "argv": sys.argv, "world_size": world,
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
    cfg_name = (f"config_rank{rank}.json" if not args.resume
                else f"config_rank{rank}_resume_e{start_epoch}_s{start_step + 1}.json")
    (log_dir / cfg_name).write_text(json.dumps(config, indent=2))
    log(f"run {run_id} | mode {args.mode} | world {world} | local batches {local_batches} | "
        f"G {G} | lr {args.lr} | k-index {k} (k={HORIZONS[k]}) | head {args.head}")
    log(f"hosts: {[(r['rank'], r['hostname'], r['gpu'], r['link']['speed']) for r in ranks]}")
    log(f"params {n_params:,} | gradient {grad_bytes / 1e6:.3f} MB ({grad_bytes / 2**20:.3f} MiB)")
    log(f"samples: train {len(train)} | val {len(val)} | test {len(test)} | "
        f"steps/epoch {steps_per_epoch} | checkpoint every {args.ckpt_every} steps")
    log(notes[0])

    step_header = ["epoch", "step", "rank", "local_batch", "t_fwd_ms", "t_bwd_ms", "t_opt_ms",
                   "t_step_ms", "loss"]
    epoch_header = ["epoch", "time_s", "samples_per_s", "train_loss", "val_loss", "val_acc",
                    "val_f1_macro", "ckpt_s"]
    scale = world / G  # 2-node loss scaling
    sync_ctx = net.no_sync if args.no_sync else contextlib.nullcontext
    if args.no_sync:
        log("NO-SYNC TIMING RUN: no gradient exchange (ranks drift apart); no checkpoints, "
            "no validation, no test")

    def make_ckpt(epoch, step, done, prog):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step, "epoch_done": done, "progress": prog,
                "best_val_loss": best_val, "best_epoch": best_epoch, "rng": rng_state(),
                "args": vars(args), "derived": derived, "rank": rank, "run_id": run_id}

    for epoch in range(start_epoch, args.epochs + 1):
        s0 = start_step if epoch == start_epoch else 0
        prog = progress if epoch == start_epoch else {"loss_sum": 0.0, "n": 0, "time": 0.0,
                                                      "ckpt_s": 0.0}
        net.train()
        g = torch.Generator()
        g.manual_seed(args.seed + epoch)
        perm_cpu = torch.randperm(len(train), generator=g)       # must be identical on every rank
        if distributed:
            digests = gather(sha256_arrays(perm_cpu.numpy()), distributed, world)
            if any(d != digests[0] for d in digests):
                raise SystemExit(f"epoch {epoch}: train permutation differs between ranks")
        perm = perm_cpu.to(dev)
        n_run = steps_per_epoch - s0
        ev = [[torch.cuda.Event(enable_timing=True) for _ in range(5)] for _ in range(n_run)]
        step_loss = torch.zeros(max(n_run, 1), device=dev)
        loss_sum = torch.tensor(prog["loss_sum"], dtype=torch.float64, device=dev)
        n_done, ckpt_s, flushed = prog["n"], 0.0, 0

        def flush_rows(upto):
            nonlocal flushed
            losses = step_loss[flushed:upto].tolist()
            rows = []
            for j in range(flushed, upto):
                e = ev[j]
                rows.append([epoch, s0 + j + 1, rank, lb,
                             round(e[1].elapsed_time(e[2]), 4), round(e[2].elapsed_time(e[3]), 4),
                             round(e[3].elapsed_time(e[4]), 4), round(e[0].elapsed_time(e[4]), 4),
                             round(losses[j - flushed], 6)])
            append_csv(steps_csv, step_header, rows)
            flushed = upto

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for j, s in enumerate(range(s0, steps_per_epoch)):
            e = ev[j]
            e[0].record()
            idx = perm[s * G:(s + 1) * G][lo:hi]
            xb, yb = train.batch(idx, k)
            e[1].record()
            with sync_ctx():                              # no_sync(): no gradient exchange
                out = net(xb)
                ce_sum = F.cross_entropy(out, yb, reduction="sum")
                loss = ce_sum * scale if distributed else ce_sum / lb
                e[2].record()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                e[3].record()
            optimizer.step()
            e[4].record()
            step_loss[j] = ce_sum.detach() / lb
            loss_sum += ce_sum.detach().double()
            n_done += lb
            if is_main and args.log_every and (s + 1) % args.log_every == 0:
                print(f"  epoch {epoch} step {s + 1}/{steps_per_epoch} loss {step_loss[j].item():.4f}",
                      flush=True)
            if (args.ckpt_every and not args.no_sync and (s + 1) % args.ckpt_every == 0
                    and s + 1 < steps_per_epoch):
                torch.cuda.synchronize()
                tc = time.perf_counter()
                flush_rows(j + 1)
                seg = tc - t0 - ckpt_s
                save_durable(make_ckpt(epoch, s + 1, False,
                                       {"loss_sum": loss_sum.item(), "n": n_done,
                                        "time": prog["time"] + seg,
                                        "ckpt_s": prog["ckpt_s"] + ckpt_s
                                        + (time.perf_counter() - tc)}),
                             rank_dir / "last.pt", keep_prev=True)
                ckpt_s += time.perf_counter() - tc
        torch.cuda.synchronize()
        train_time = prog["time"] + (time.perf_counter() - t0 - ckpt_s)
        flush_rows(n_run)

        per_rank = gather({"loss_sum": loss_sum.item(), "n": n_done, "time": train_time},
                          distributed, world)
        info = None
        if is_main:
            time_s = max(pr["time"] for pr in per_rank)
            train_loss = sum(pr["loss_sum"] for pr in per_rank) / sum(pr["n"] for pr in per_rank)
        if is_main and args.no_sync:
            nan = float("nan")
            info = {"time_s": time_s, "train_loss": train_loss, "improved": False,
                    "best_val": best_val, "best_epoch": best_epoch,
                    "vm": {"loss": nan, "acc": nan, "f1_macro": nan}}
        elif is_main:
            vm = evaluate(model, val, k, args.head, args.eval_batch)
            if not math.isfinite(vm["loss"]):
                print(f"WARNING: epoch {epoch} val loss is {vm['loss']}", flush=True)
            improved = vm["loss"] < best_val
            if improved:
                best_val, best_epoch = vm["loss"], epoch
            if improved or not (ckpt_dir / "best.pt").exists():
                save_durable(make_ckpt(epoch, steps_per_epoch, True,
                                       {"loss_sum": 0.0, "n": 0, "time": 0.0, "ckpt_s": 0.0}),
                             ckpt_dir / "best.pt")
            info = {"time_s": time_s, "train_loss": train_loss, "vm": vm, "improved": improved,
                    "best_val": best_val, "best_epoch": best_epoch}
        if distributed:                                  # val result + best-so-far to all ranks
            box = [info]
            dist.broadcast_object_list(box, src=0)
            info = box[0]
            best_val, best_epoch = info["best_val"], info["best_epoch"]
        vm = info["vm"]
        append_csv(epochs_csv, epoch_header,
                   [[epoch, round(info["time_s"], 3), round(steps_per_epoch * G / info["time_s"], 2),
                     round(info["train_loss"], 6), round(vm["loss"], 6), round(vm["acc"], 6),
                     round(vm["f1_macro"], 6), round(prog["ckpt_s"] + ckpt_s, 3)]])
        # best.pt (rank 0) is written before last.pt, so last.pt never names a best epoch
        # that is not on disk
        if not args.no_sync:
            save_durable(make_ckpt(epoch, steps_per_epoch, True,
                                   {"loss_sum": 0.0, "n": 0, "time": 0.0, "ckpt_s": 0.0}),
                         rank_dir / "last.pt", keep_prev=True)
        log(f"epoch {epoch}/{args.epochs} | {info['time_s']:.1f} s | "
            f"{steps_per_epoch * G / info['time_s']:.0f} samples/s | "
            f"train_loss {info['train_loss']:.4f} | val_loss {vm['loss']:.4f} | "
            f"val_acc {vm['acc']:.4f} | val_f1 {vm['f1_macro']:.4f}{' | best' if info['improved'] else ''}")

    # ---- test (rank 0 evaluates best.pt; every rank writes the result to its own logs/, so a
    # job queue on any node can see that the run is finished)
    result = None
    if args.no_sync:                                      # timing-only run: completion marker
        result = {"run_id": run_id, "no_sync": True,
                  "note": "timing-only run (--no-sync): no gradient exchange, no validation, no test"}
    elif is_main and not (ckpt_dir / "best.pt").exists():
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
                  "n_test": tm["n"], "evaluated_on_rank": 0}
        print(f"test (best.pt, epoch {best['epoch']}): acc {tm['acc']:.4f} | "
              f"f1_macro {tm['f1_macro']:.4f} | n {tm['n']}", flush=True)
    if distributed:
        box = [result]
        dist.broadcast_object_list(box, src=0)
        result = box[0]
    if result is not None:
        tmp = log_dir / f"test.json.tmp{rank}"            # per rank: safe on a shared disk
        with open(tmp, "w") as f:
            f.write(json.dumps(result, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, log_dir / "test.json")
        fsync_dir(log_dir)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
