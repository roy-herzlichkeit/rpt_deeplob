#!/usr/bin/env python3
"""Choose balanced local batch sizes for 2-node DDP from the two nodes' bench_batch.py CSVs.

In synchronous DDP every step waits for the slower rank, so the compute part of a step is
    max(t_A(b_A), t_B(G - b_A)).
t_X(b) is node X's step_ms_median at batch b, linearly interpolated between measured batch sizes
(the all_reduce cost is the same for every split, so it does not change the choice).
The script prints every split, marks the best one, compares it with the equal split, and lists
the batch sizes worth measuring directly before using the split (interpolated values are estimates).

Example:
  python scripts/make_split.py --a results/bench_herzlichkeit-ubuntu_cudnn-on.csv \
                               --b results/bench_pritam_cudnn-on.csv --global-batch 64
"""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(path):
    pts = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["step_ms_median"] != "OOM":
                pts.append((int(r["batch"]), float(r["step_ms_median"])))
    pts.sort()
    if len(pts) < 2:
        raise SystemExit(f"{path}: need at least 2 measured batch sizes")
    return pts


def interp(pts, b):
    """Linear interpolation of step time at batch b; None outside the measured range."""
    if b < pts[0][0] or b > pts[-1][0]:
        return None
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        if b0 <= b <= b1:
            return t0 if b1 == b0 else t0 + (t1 - t0) * (b - b0) / (b1 - b0)
    return pts[-1][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", default=str(REPO / "results" / "bench_herzlichkeit-ubuntu_cudnn-on.csv"),
                    help="bench CSV of rank 0 (node A)")
    ap.add_argument("--b", default=str(REPO / "results" / "bench_pritam_cudnn-on.csv"),
                    help="bench CSV of rank 1 (node B)")
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--out", default=None, help="optional JSON file for the result")
    args = ap.parse_args()

    G = args.global_batch
    a, b = load(args.a), load(args.b)
    rows = []
    for ba in range(1, G):
        bb = G - ba
        ta, tb = interp(a, ba), interp(b, bb)
        if ta is None or tb is None:
            continue
        rows.append({"local_batches": [ba, bb], "t_a_ms": ta, "t_b_ms": tb,
                     "step_ms": max(ta, tb), "idle_ms": abs(ta - tb),
                     "measured": ba in dict(a) and bb in dict(b)})
    if not rows:
        raise SystemExit(f"no split of {G} has both local batches inside the measured ranges")

    best = min(rows, key=lambda r: (r["step_ms"], r["idle_ms"]))
    equal = next((r for r in rows if r["local_batches"] == [G // 2, G - G // 2]), None)

    print(f"A: {args.a}\nB: {args.b}\nG = {G}   (t = step_ms_median, interpolated; * = both measured)")
    print(f"{'A,B':>7} {'t_A ms':>8} {'t_B ms':>8} {'max ms':>8} {'idle ms':>8}")
    for r in rows:
        mark = " <- best" if r is best else (" <- equal" if r is equal else "")
        print(f"{r['local_batches'][0]:>3},{r['local_batches'][1]:<3} {r['t_a_ms']:8.3f} {r['t_b_ms']:8.3f} "
              f"{r['step_ms']:8.3f} {r['idle_ms']:8.3f}{'*' if r['measured'] else ' '}{mark}")
    print(f"\nbest split: --local-batches {best['local_batches'][0]},{best['local_batches'][1]}"
          f"   compute step ~{best['step_ms']:.3f} ms")
    if equal:
        print(f"equal split {equal['local_batches']}: ~{equal['step_ms']:.3f} ms -> best is "
              f"{equal['step_ms'] / best['step_ms']:.3f}x faster in compute (all_reduce not included)")
    if not best["measured"]:
        print(f"note: estimate is interpolated. Measure it directly before use:\n"
              f"  node A: python scripts/bench_batch.py --node A --batches {best['local_batches'][0]} --out <file>\n"
              f"  node B: python scripts/bench_batch.py --node B --batches {best['local_batches'][1]} --out <file>")
    if args.out:
        Path(args.out).write_text(json.dumps({"global_batch": G, "a_csv": args.a, "b_csv": args.b,
                                              "best": best, "equal": equal}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
