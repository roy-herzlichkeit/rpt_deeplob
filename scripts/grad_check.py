#!/usr/bin/env python3
"""Check that train.py's 2-node loss scaling gives the exact mean gradient of the global batch.

On every rank, for a few global batches (same permutation rule as train.py):
  reference : the full global batch of G samples on one GPU, mean cross-entropy, backward
  DDP       : rank r takes its slice (local batches in rank order), loss = CE_sum * world / G,
              backward through DDP (which averages over world)
and compares the DDP gradient with the reference gradient.

A second reference ("split") computes the same slices on one GPU without DDP, each slice in its
own forward pass, summing CE_sum / G.
PASS needs, for every step and rank (fp32 rounding tolerance --tol):
  BN-eval  (running stats, split-independent forward): DDP == split == full batch
  BN-train (local BatchNorm, as in train.py):          DDP == split
In BN-train the full-batch difference is printed for information only: with local BatchNorm each
rank normalises with its own batch statistics, so the gradient depends on the split by design
(at initialisation this difference is large, mostly in conv weights before BatchNorm layers).

Run (2 nodes, same as train.py):
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=R --master_addr=192.168.10.1 \
      --master_port=29500 scripts/grad_check.py --backend nccl --local-batches 40,24
Local test on one GPU (gloo, loopback):
  torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29513 \
      scripts/grad_check.py --backend gloo --local-batches 40,24
"""
import argparse
import copy
import datetime
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as tr  # noqa: E402


def grads(model):
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def set_bn_eval(model, flag):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train(not flag)


def fresh(base, bn_eval):
    m = copy.deepcopy(base)
    m.lstm.flatten_parameters()          # deepcopy breaks cuDNN's contiguous LSTM weights
    set_bn_eval(m, bn_eval)
    return m


def rel(a, b):
    return (a - b).norm().item() / b.norm().item()


def compare(base, train_split, perm, G, bounds, rank, k, world, bn_eval, step):
    idx = perm[step * G:(step + 1) * G]
    # reference 1: whole global batch on one GPU, mean loss
    ref = fresh(base, bn_eval)
    xb, yb = train_split.batch(idx, k)
    F.cross_entropy(ref(xb), yb).backward()
    g_ref = grads(ref)
    # reference 2: same split as DDP but on one GPU without DDP: each slice is a separate
    # forward (own BatchNorm statistics), sum over slices of CE_sum / G
    split = fresh(base, bn_eval)
    for a, b in bounds:
        xs, ys = train_split.batch(idx[a:b], k)
        (F.cross_entropy(split(xs), ys, reduction="sum") / G).backward()
    g_split = grads(split)
    # DDP: local slice, CE_sum * world / G, averaged by DDP
    local = fresh(base, bn_eval)
    net = DDP(local, device_ids=[0], broadcast_buffers=False)
    lo, hi = bounds[rank]
    xl, yl = train_split.batch(idx[lo:hi], k)
    (F.cross_entropy(net(xl), yl, reduction="sum") * world / G).backward()
    g_ddp = grads(local)
    return {"rel_full": rel(g_ddp, g_ref), "rel_split": rel(g_ddp, g_split),
            "max_abs_full": (g_ddp - g_ref).abs().max().item(), "grad_norm": g_ref.norm().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--local-batches", default="40,24", help="rank order, must sum to G")
    ap.add_argument("--k-index", type=int, default=3, choices=range(5))
    ap.add_argument("--head", choices=["softmax", "logits"], default="logits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=3, help="number of global batches to check")
    ap.add_argument("--tol", type=float, default=1e-4, help="max relative error for PASS")
    args = ap.parse_args()

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world < 2:
        raise SystemExit("run with torchrun and >= 2 processes")
    lb = [int(v) for v in args.local_batches.split(",")]
    G = args.global_batch
    if len(lb) != world or sum(lb) != G or min(lb) <= 0:
        raise SystemExit(f"--local-batches {lb}: need {world} positive values that sum to {G}")
    bounds = [(sum(lb[:r]), sum(lb[:r + 1])) for r in range(world)]

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    # correctness check: fp32, no TF32, deterministic cuDNN so the only differences are rounding
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    extra = {"device_id": dev} if args.backend == "nccl" else {}
    dist.init_process_group(args.backend, timeout=datetime.timedelta(minutes=10), **extra)

    x, y = tr.load_file(tr.TRAIN_FILE)
    n_tr = int(x.shape[0] * (1 - tr.VAL_FRAC))
    train_split = tr.Split(torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev),
                           tr.windows_in(0, n_tr))
    torch.manual_seed(args.seed)
    base = tr.DeepLOB(3, head=args.head).to(dev).train()
    g = torch.Generator()
    g.manual_seed(args.seed + 1)                       # same rule as train.py, epoch 1
    perm = torch.randperm(len(train_split), generator=g).to(dev)

    results = []
    for s in range(args.steps):
        for bn_eval in (True, False):
            r = compare(base, train_split, perm, G, bounds, rank, args.k_index, world, bn_eval, s)
            results.append((s + 1, "eval" if bn_eval else "train", r))
    out = [None] * world
    dist.all_gather_object(out, {"rank": rank, "host": socket.gethostname(), "lb": lb[rank],
                                 "results": results})
    if rank == 0:
        ok = True
        print(f"grad_check | backend {args.backend} | world {world} | G {G} | local batches {lb} | "
              f"k-index {args.k_index} | head {args.head} | tol {args.tol:g}")
        for o in out:
            for step, bn, r in o["results"]:
                # DDP must equal the same split computed without DDP, in both BN modes;
                # it must equal the full-batch gradient only when BN uses running stats (eval)
                passed = r["rel_split"] < args.tol and (bn == "train" or r["rel_full"] < args.tol)
                ok &= passed
                print(f"  rank {o['rank']} ({o['host']}, lb {o['lb']}) step {step} BN-{bn:5s}: "
                      f"rel vs split {r['rel_split']:.2e} | rel vs full batch {r['rel_full']:.2e}"
                      f"{' (local BN effect, info)' if bn == 'train' else ''} | |g| {r['grad_norm']:.2e}"
                      f"  {'PASS' if passed else 'FAIL'}")
        print("RESULT:", "PASS" if ok else "FAIL")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
