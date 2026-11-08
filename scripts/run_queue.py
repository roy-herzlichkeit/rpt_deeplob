#!/usr/bin/env python3
"""Run the training runs listed in configs/queue.txt, in order, and survive restarts.

queue.txt: one run per line, "<run_id>: <train.py arguments>"; blank lines and # comments are
ignored. Do not put --run-id or --resume in the arguments (this script adds them). Example:
    g64_balanced_43_21: --mode balanced --local-batches 43,21 --epochs 5
    g64_single:         --mode single --epochs 5

For each run, on THIS node:
  logs/<run_id>/test.json exists              -> finished, skip it
  checkpoints/<run_id>/rank<r>/last.pt (or .prev) exists, no test.json
                                              -> start it with --resume
  neither                                     -> start it fresh; leftovers of an attempt that died
                                                 before its first checkpoint are moved aside
                                                 (logs/<run_id>.stale-<time>) first
--mode single runs only on node rank 0 (other nodes skip them). 2-node runs start torchrun with
this node's --node-rank, so start run_queue.py on BOTH nodes with the same queue.txt.
The queue stops at the first run that fails (for example after the peer node died); start
run_queue.py again on both nodes to resume from the checkpoints.

Example (node A; node B uses --node-rank 1 and its own --ifname):
  python scripts/run_queue.py --node-rank 0 --ifname eno1 --dry-run
"""
import argparse
import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def read_queue(path):
    runs, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise SystemExit(f"{path}:{n}: expected '<run_id>: <train.py arguments>'")
        run_id, rest = (x.strip() for x in line.split(":", 1))
        argv = shlex.split(rest)
        if not run_id or any(a in ("--run-id", "--resume") or a.startswith(("--run-id=", "--resume="))
                             for a in argv):
            raise SystemExit(f"{path}:{n}: need a run_id and no --run-id/--resume in the arguments")
        if "--mode" not in argv:
            raise SystemExit(f"{path}:{n}: --mode is required")
        if run_id in seen:
            raise SystemExit(f"{path}:{n}: run_id '{run_id}' appears twice")
        seen.add(run_id)
        runs.append((run_id, argv, argv[argv.index("--mode") + 1]))
    return runs


def state(run_id, rank):
    log_dir, rank_dir = REPO / "logs" / run_id, REPO / "checkpoints" / run_id / f"rank{rank}"
    if (log_dir / "test.json").exists():
        return "done"
    if any((rank_dir / n).exists() for n in ("last.pt", "last.pt.prev")):
        return "resume"
    if log_dir.exists() or (REPO / "checkpoints" / run_id).exists():
        return "stale"
    return "new"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--queue", default=str(REPO / "configs" / "queue.txt"))
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--nnodes", type=int, default=2)
    ap.add_argument("--master-addr", default="192.168.10.1")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--ifname", default=None,
                    help="Ethernet interface for NCCL/Gloo on this node (sets NCCL_SOCKET_IFNAME, "
                         "GLOO_SOCKET_IFNAME, NCCL_IB_DISABLE=1)")
    ap.add_argument("--dry-run", action="store_true", help="only print what would run")
    args = ap.parse_args()

    runs = read_queue(args.queue)
    env = dict(os.environ)
    if args.ifname:
        env.update(NCCL_SOCKET_IFNAME=args.ifname, GLOO_SOCKET_IFNAME=args.ifname, NCCL_IB_DISABLE="1")
    print(f"queue {args.queue}: {len(runs)} runs | node rank {args.node_rank}", flush=True)

    for i, (run_id, argv, mode) in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}] {run_id}"
        if mode == "single" and args.node_rank != 0:
            print(f"{tag}: single-GPU run, node rank 0 only -> skip on this node", flush=True)
            continue
        st = state(run_id, 0 if mode == "single" else args.node_rank)
        if st == "done":
            print(f"{tag}: test.json exists -> skip", flush=True)
            continue
        train = [str(REPO / "scripts" / "train.py"), "--run-id", run_id, *argv]
        if st == "resume":
            train.append("--resume")
        if mode == "single":
            cmd = [PY, *train]
        else:
            cmd = [PY, "-m", "torch.distributed.run", f"--nnodes={args.nnodes}", "--nproc_per_node=1",
                   f"--node_rank={args.node_rank}", f"--master_addr={args.master_addr}",
                   f"--master_port={args.master_port}", *train]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"{tag}: {st} -> {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        if st == "stale":                     # died before its first checkpoint: start over
            for d in (REPO / "logs" / run_id, REPO / "checkpoints" / run_id):
                if d.exists():
                    d.rename(d.with_name(f"{d.name}.stale-{stamp}"))
                    print(f"{tag}: moved {d} aside", flush=True)
        console = REPO / "logs" / f"queue_{run_id}_node{args.node_rank}.log"
        console.parent.mkdir(parents=True, exist_ok=True)
        with open(console, "a") as log:
            log.write(f"\n===== {stamp} {st}: {shlex.join(cmd)}\n")
            log.flush()
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = p.wait()
        if rc != 0 or not (REPO / "logs" / run_id / "test.json").exists():
            raise SystemExit(f"{tag}: failed (exit code {rc}). The queue stops here; start "
                             f"run_queue.py again on all nodes to resume.")
        print(f"{tag}: finished", flush=True)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
