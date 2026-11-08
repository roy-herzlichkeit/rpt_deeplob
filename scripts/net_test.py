"""2-node all_reduce test (launch with torchrun).

test 1: correctness, tensor [rank+1] all_reduce SUM (2 ranks -> 3.0)
test 2: latency, 150,000 fp32 (0.6 MB), 20 warm-up + 200 timed runs
test 3: bandwidth, 25,000,000 fp32 (100 MB), 3 warm-up + 10 timed runs
"""
import argparse
import datetime
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist


def timed_allreduce(t, warmup, iters):
    for _ in range(warmup):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def p95(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    args = ap.parse_args()

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    extra = {"device_id": dev} if args.backend == "nccl" else {}
    dist.init_process_group(backend=args.backend,
                            timeout=datetime.timedelta(seconds=120), **extra)
    rank, world = dist.get_rank(), dist.get_world_size()
    tag = f"[rank {rank} | {socket.gethostname()} | {args.backend}]"
    print(f"{tag} world_size={world} gpu={torch.cuda.get_device_name(0)} "
          f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
          flush=True)

    # test 1: correctness
    t = torch.tensor([float(rank + 1)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = world * (world + 1) / 2
    ok = abs(t.item() - expected) < 1e-6
    print(f"{tag} test1 all_reduce SUM: got {t.item():.1f}, expected {expected:.1f} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)

    # test 2: small-message latency (0.6 MB)
    small = torch.ones(150_000, dtype=torch.float32, device=dev)
    ts = timed_allreduce(small, warmup=20, iters=200)
    print(f"{tag} test2 0.6 MB x200: median {statistics.median(ts) * 1e3:.3f} ms, "
          f"p95 {p95(ts) * 1e3:.3f} ms", flush=True)

    # test 3: large-message bandwidth (100 MB)
    big = torch.ones(25_000_000, dtype=torch.float32, device=dev)
    tb = timed_allreduce(big, warmup=3, iters=10)
    med = statistics.median(tb)
    mb = big.numel() * big.element_size() / 1e6
    print(f"{tag} test3 {mb:.0f} MB x10: median {med * 1e3:.1f} ms, "
          f"{mb / med:.1f} MB/s (algbw)", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
