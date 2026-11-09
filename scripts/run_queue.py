#!/usr/bin/env python3
"""Run the training runs listed in configs/queue.txt, in order, and survive restarts.

queue.txt: one run per line, "<run_id>: <train.py arguments>"; blank lines and # comments are
ignored. Do not put --run-id or --resume in the arguments (this script adds them).
A single-GPU run (--mode single) must name the node that runs it: --node A or --node B (this
option is read by run_queue.py and not passed to train.py). Example:
    g64_balanced_43_21: --mode balanced --local-batches 43,21 --epochs 5
    g64_single_A:       --mode single --node A --epochs 5

For each run, on THIS node:
  logs/<run_id>/test.json exists              -> finished, skip it
  checkpoints/<run_id>/rank<r>/last.pt (or .prev) exists, no test.json
                                              -> start it with --resume
  neither                                     -> start it fresh; leftovers of an attempt that died
                                                 before its first checkpoint are moved aside
                                                 (logs/<run_id>.stale-<time>) first
R6 slow-down: a queue line may carry "--slow A:<F>" (also B:<F>, repeatable). It is read by
run_queue.py: only the named node passes --slow-factor <F> to train.py; the other node runs the
same line unchanged. The queue file stays shared.
This node's name is --node if given, otherwise it follows --node-rank (0 = A, 1 = B), so node B's
usual command (--node-rank 1, no --node) needs no change.
Single-GPU runs run only on the node named by their --node (give run_queue.py --node A or B);
the other node skips them without waiting, and --node-rank does not matter for them.
2-node runs start torchrun with this node's --node-rank, so start run_queue.py on BOTH nodes
with the same queue.txt.
The queue stops at the first run that fails (for example after the peer node died); start
run_queue.py again on both nodes to resume from the checkpoints.
Link check (2-node runs): before each run, /sys/class/net/<--ifname>/speed must be --link-speed
(default 1000 Mb/s) or the queue stops. After the run the speed is read again; if it changed,
"link_changed" is written into logs/<run_id>/config_rank<node rank>.json.

Examples:
  node A, 2-node queue:  python scripts/run_queue.py --queue configs/queue_main.txt --node A --node-rank 0 --ifname eno1
  node B, 2-node queue:  python scripts/run_queue.py --queue configs/queue_main.txt --node B --node-rank 1 --ifname <iface>
  node A, single queue:  python scripts/run_queue.py --queue configs/queue_single_A.txt --node A
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
NODES = ("A", "B")


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
        mode = argv[argv.index("--mode") + 1]
        slow = {}
        while "--slow" in argv:                    # run_queue option: per-node slow factor
            i = argv.index("--slow")
            val = argv[i + 1] if i + 1 < len(argv) else ""
            del argv[i:i + 2]
            who, _, fac = val.partition(":")
            try:
                fac = float(fac)
            except ValueError:
                fac = -1.0
            if who not in NODES or fac < 1.0:
                raise SystemExit(f"{path}:{n}: --slow needs <node>:<factor>, node A or B, factor >= 1")
            slow[who] = fac
        if any(a == "--slow-factor" or a.startswith("--slow-factor=") for a in argv):
            raise SystemExit(f"{path}:{n}: use --slow A:<F> in the queue, not --slow-factor")
        node = None
        if "--node" in argv:                       # run_queue option, not a train.py argument
            i = argv.index("--node")
            node = argv[i + 1] if i + 1 < len(argv) else ""
            del argv[i:i + 2]
            if node not in NODES:
                raise SystemExit(f"{path}:{n}: --node must be one of {NODES}")
        if mode == "single" and node is None:
            raise SystemExit(f"{path}:{n}: a single-GPU run needs --node A or --node B")
        if mode != "single" and node is not None:
            raise SystemExit(f"{path}:{n}: --node is only for single-GPU runs")
        if run_id in seen:
            raise SystemExit(f"{path}:{n}: run_id '{run_id}' appears twice")
        seen.add(run_id)
        runs.append((run_id, argv, mode, node, slow))
    return runs


def link_speed(iface):
    """Current link speed of iface in Mb/s, or None if it cannot be read (link down, no iface)."""
    try:
        return int(Path(f"/sys/class/net/{iface}/speed").read_text().strip())
    except (OSError, ValueError):
        return None


def mark_link_changed(run_id, node_rank, iface, before, after):
    """Record a link speed change in this node's config file of the run."""
    entry = {"iface": iface, "before_mbps": before, "after_mbps": after,
             "checked": datetime.datetime.now().isoformat(timespec="seconds")}
    log_dir = REPO / "logs" / run_id
    cfgs = sorted(log_dir.glob(f"config_rank{node_rank}*.json")) if log_dir.exists() else []
    if not cfgs:                       # run died before writing its config: separate file
        log_dir.mkdir(parents=True, exist_ok=True)
        cfgs = [log_dir / f"link_changed_rank{node_rank}.json"]
    for path in cfgs:
        data = json.loads(path.read_text()) if path.exists() else {}
        data["link_changed"] = entry
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
    return cfgs


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
    ap.add_argument("--node", choices=NODES, default=None,
                    help="this node's name; needed if the queue has single-GPU runs")
    ap.add_argument("--node-rank", type=int, default=0, help="torchrun node rank for 2-node runs")
    ap.add_argument("--nnodes", type=int, default=2)
    ap.add_argument("--master-addr", default="192.168.10.1")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--ifname", default=None,
                    help="Ethernet interface for NCCL/Gloo on this node (sets NCCL_SOCKET_IFNAME, "
                         "GLOO_SOCKET_IFNAME, NCCL_IB_DISABLE=1)")
    ap.add_argument("--link-speed", type=int, default=1000,
                    help="required link speed in Mb/s on --ifname before each 2-node run")
    ap.add_argument("--dry-run", action="store_true", help="only print what would run")
    args = ap.parse_args()

    runs = read_queue(args.queue)
    if args.node is None and args.node_rank in (0, 1):  # 2-node default: rank 0 = A, rank 1 = B
        if not any(m == "single" for _, _, m, _, _ in runs):
            args.node = NODES[args.node_rank]
    if any(m == "single" for _, _, m, _, _ in runs) and args.node is None:
        raise SystemExit("this queue has single-GPU runs: give this node's name with --node A or B")
    if any(m != "single" for _, _, m, _, _ in runs) and not args.ifname:
        raise SystemExit("this queue has 2-node runs: give the Ethernet interface with --ifname "
                         "(used for NCCL/Gloo and for the link speed check)")
    env = dict(os.environ)
    if args.ifname:
        env.update(NCCL_SOCKET_IFNAME=args.ifname, GLOO_SOCKET_IFNAME=args.ifname, NCCL_IB_DISABLE="1")
    print(f"queue {args.queue}: {len(runs)} runs | node {args.node} | node rank {args.node_rank}",
          flush=True)

    for i, (run_id, argv, mode, node, slow) in enumerate(runs, 1):
        tag = f"[{i}/{len(runs)}] {run_id}"
        if mode == "single" and node != args.node:
            print(f"{tag}: single-GPU run for node {node} -> skip on node {args.node}", flush=True)
            continue
        st = state(run_id, 0 if mode == "single" else args.node_rank)
        if st == "done":
            print(f"{tag}: test.json exists -> skip", flush=True)
            continue
        train = [str(REPO / "scripts" / "train.py"), "--run-id", run_id, *argv]
        if st == "resume":
            train.append("--resume")
        if args.node in slow:                      # R6: only the named node is slowed
            train += ["--slow-factor", f"{slow[args.node]:g}"]
        if mode == "single":
            cmd = [PY, *train]
        else:
            cmd = [PY, "-m", "torch.distributed.run", f"--nnodes={args.nnodes}", "--nproc_per_node=1",
                   f"--node_rank={args.node_rank}", f"--master_addr={args.master_addr}",
                   f"--master_port={args.master_port}", *train]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"{tag}: {st} -> {shlex.join(cmd)}", flush=True)
        speed_before = None
        if mode != "single":
            speed_before = link_speed(args.ifname)
            ok = speed_before == args.link_speed
            print(f"{tag}: link {args.ifname} = {speed_before} Mb/s "
                  f"({'OK' if ok else f'NOT {args.link_speed}'})", flush=True)
            if not ok and not args.dry_run:
                raise SystemExit(
                    f"{tag}: STOP - link {args.ifname} is {speed_before} Mb/s, not "
                    f"{args.link_speed} Mb/s. Fix the cable/link (check "
                    f"/sys/class/net/{args.ifname}/speed on both nodes), then start run_queue.py "
                    f"again on both nodes. The other node will stop when it cannot reach this one.")
        if args.dry_run:
            continue
        if st == "stale":                     # died before its first checkpoint: start over
            for d in (REPO / "logs" / run_id, REPO / "checkpoints" / run_id):
                if d.exists():
                    d.rename(d.with_name(f"{d.name}.stale-{stamp}"))
                    print(f"{tag}: moved {d} aside", flush=True)
        console = REPO / "logs" / (f"queue_{run_id}_node{args.node}.log" if mode == "single"
                                   else f"queue_{run_id}_node{args.node_rank}.log")
        console.parent.mkdir(parents=True, exist_ok=True)
        with open(console, "a") as log:
            log.write(f"\n===== {stamp} {st}: {shlex.join(cmd)}\n")
            log.flush()
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()                   # keep the console log if the machine dies
            rc = p.wait()
        if mode != "single":
            speed_after = link_speed(args.ifname)
            if speed_after != speed_before:
                files = mark_link_changed(run_id, args.node_rank, args.ifname, speed_before,
                                          speed_after)
                print(f"{tag}: WARNING link_changed: {args.ifname} {speed_before} -> {speed_after} "
                      f"Mb/s during the run; recorded in {', '.join(str(p) for p in files)}",
                      flush=True)
        if rc != 0 or not (REPO / "logs" / run_id / "test.json").exists():
            raise SystemExit(f"{tag}: failed (exit code {rc}). The queue stops here; start "
                             f"run_queue.py again on all nodes to resume.")
        print(f"{tag}: finished", flush=True)
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
