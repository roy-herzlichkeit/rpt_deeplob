#!/usr/bin/env python3
"""Detailed test-set evaluation of a run's best checkpoint.

Loads checkpoints/<run_id>/best.pt (written by rank 0 of train.py), evaluates it on the test set
(Test CF_7/8/9, windows inside each file, same as train.py) and writes
results/eval_<run_id>.json with per-class precision / recall / F1 / support, macro and weighted
averages, accuracy and the confusion matrix (raw counts and row-normalised = recall per class).
The head and k-index are taken from the checkpoint's own arguments.

  python scripts/eval_report.py                 # m_bal128_s1
  python scripts/eval_report.py --run-id m_eq128_s1
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as tr  # noqa: E402


@torch.no_grad()
def predict(model, split, k, bs):
    model.eval()
    preds = []
    for i in range(0, len(split), bs):
        idx = torch.arange(i, min(i + bs, len(split)), device=split.x.device)
        xb, _ = split.batch(idx, k)
        preds.append(model(xb).argmax(1))
    return split.y[split.ends, k].cpu().numpy(), torch.cat(preds).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-id", default="m_bal128_s1")
    ap.add_argument("--eval-batch", type=int, default=1024)
    ap.add_argument("--out", default=None, help="default: results/eval_<run_id>.json")
    args = ap.parse_args()

    ckpt_path = tr.REPO / "checkpoints" / args.run_id / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found (best.pt is kept on node A / rank 0)")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_args = ckpt["args"]
    k, head = run_args["k_index"], run_args["head"]

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    torch.backends.cudnn.allow_tf32 = False            # same numerics as train.py
    torch.backends.cuda.matmul.allow_tf32 = False
    _, _, test, data_info = tr.build_data(dev)
    model = tr.DeepLOB(3, head=head).to(dev)
    model.load_state_dict(ckpt["model"])
    y_true, y_pred = predict(model, test, k, args.eval_batch)

    labels = [0, 1, 2]
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro",
                                                     zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                     average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred))
    names = tr.CLASS_NAMES

    test_json = tr.REPO / "logs" / args.run_id / "test.json"
    ref = json.loads(test_json.read_text()) if test_json.exists() else None
    result = {
        "run_id": args.run_id,
        "checkpoint": str(ckpt_path.relative_to(tr.REPO)),
        "best_epoch": ckpt["epoch"], "best_val_loss": ckpt["best_val_loss"],
        "k_index": k, "horizon_k": tr.HORIZONS[k], "head": head,
        "n_test": int(len(y_true)), "test_files": [f["file"] for f in data_info["test_files"]],
        "class_names": names,
        "accuracy": acc,
        "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f1[i]), "support": int(sup[i]),
                                 "predicted": int(cm[:, i].sum())} for i in labels},
        "macro_avg": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
        "weighted_avg": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
        "confusion_matrix": {"rows": "true class", "cols": "predicted class", "labels": names,
                             "counts": cm.tolist(),
                             "row_normalised": np.round(cm / cm.sum(1, keepdims=True), 6).tolist()},
        "check_against_test_json": None if ref is None else {
            "test_json_accuracy": ref["accuracy"], "test_json_f1_macro": ref["f1_macro"],
            "same_confusion_matrix": ref["confusion_matrix"] == cm.tolist()},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else tr.REPO / "results" / f"eval_{args.run_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{args.run_id} best.pt (epoch {ckpt['epoch']}): accuracy {acc:.4f} | macro F1 {fm:.4f} | n {len(y_true)}")
    for i in labels:
        print(f"  {names[i]:10} precision {p[i]:.4f}  recall {r[i]:.4f}  f1 {f1[i]:.4f}  support {sup[i]}")
    print("  confusion matrix (rows true, cols predicted):", cm.tolist())
    if ref is not None:
        print(f"  test.json: accuracy {ref['accuracy']:.4f} | macro F1 {ref['f1_macro']:.4f} | "
              f"same confusion matrix: {ref['confusion_matrix'] == cm.tolist()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
