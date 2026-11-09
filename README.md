# rpt_deeplob

Two-node data-parallel training of DeepLOB on the FI-2010 limit order book dataset, on two
laptops with different GPUs connected by a direct 1 Gbit/s Ethernet cable.

## Goal

Measure how PyTorch DDP behaves on two unequal GPUs, and whether splitting the global batch
unevenly between the nodes ("balanced") is faster than an equal split, without changing the
model's results:

- node A: RTX 4060 Laptop (8 GB), `192.168.10.1`, interface `eno1`, rank 0
- node B: RTX 3060 Laptop (6 GB), `192.168.10.2`, rank 1
- backend NCCL over the Ethernet link (sockets), fp32 only, `cudnn.benchmark` on by default
- model: DeepLOB from the authors' PyTorch notebook; label horizon k=50 (`--k-index 3`)
- both global-batch modes give the exact mean gradient of the global batch
  (`CE_sum / G * world_size`, checked by `scripts/grad_check.py`); BatchNorm stays local per rank

## Setup (each node)

```bash
git clone https://github.com/roy-herzlichkeit/rpt_deeplob.git && cd rpt_deeplob
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # first line: torch from the cu128 index
# data (not in git): FI-2010 from the DeepLOB repository, unzipped into data/
mkdir -p data && cd data && wget https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip && unzip -n data.zip && cd ..
python scripts/env_check.py              # GPU, CUDA, NCCL check
```

`data/cache/` (parsed `.npy` arrays) is created on the first run. Before 2-node runs, the link
must be up at 1000 Mb/s on both nodes (`cat /sys/class/net/<iface>/speed`).

## Launch

One node (single GPU):

```bash
.venv/bin/python scripts/train.py --mode single --global-batch 128 --epochs 12 --run-id my_run
```

Two nodes (start node A first; node B within about a minute; same arguments on both):

```bash
# node A (rank 0)
NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=1 \
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=192.168.10.1 --master_port=29500 \
  scripts/train.py --mode balanced --global-batch 128 --local-batches 75,53 --epochs 12 --run-id my_run
# node B (rank 1): the same with --node_rank=1 and its own interface name in the two *_IFNAME variables
```

Useful `train.py` options: `--mode single|equal|balanced`, `--local-batches A,B` (rank order),
`--resume` (continue at the saved epoch and step), `--max-steps N` (per epoch),
`--cudnn-benchmark on|off`, `--head logits|softmax`, `--no-sync` (timing only: no gradient
exchange), `--slow-factor F` (R6 test: this rank computes F times slower). Every rank saves
`checkpoints/<run_id>/rank<r>/last.pt` every 1000 steps and at each epoch end; rank 0 keeps
`best.pt` (lowest validation loss) and evaluates it on the test set at the end.

## Run queues

`scripts/run_queue.py` runs the lines of a queue file in order, skips runs that have
`logs/<run_id>/test.json`, resumes runs that have a checkpoint, and stops at the first failure.
Before each 2-node run it checks that the link is at 1000 Mb/s (and records `link_changed` in the
run's config if the speed changed during the run). Start it on both nodes with the same file:

```bash
# node A
.venv/bin/python scripts/run_queue.py --queue configs/queue_main.txt --node-rank 0 --ifname eno1
# node B
.venv/bin/python scripts/run_queue.py --queue configs/queue_main.txt --node-rank 1 --ifname <iface>
# single-GPU queues (no peer)
.venv/bin/python scripts/run_queue.py --queue configs/queue_single_A.txt --node A
.venv/bin/python scripts/run_queue.py --queue configs/queue_single_B.txt --node B
# preview without running anything
.venv/bin/python scripts/run_queue.py --queue configs/queue_main.txt --node-rank 0 --ifname eno1 --dry-run
```

Queue line format: `<run_id>: <train.py arguments>`. Extra options read by `run_queue.py`:
`--node A|B` (single-GPU runs: which node runs it) and `--slow A:<F>` (only node A passes
`--slow-factor F`; node B runs the same line unchanged).

| queue file | runs | what it runs |
|---|---|---|
| `configs/queue_smoke.txt` | 7 | 2-node smoke test: equal vs balanced at G=64/128/256, 300 steps each, plus `ctl_bal64` (2 full epochs) |
| `configs/queue_diag.txt` | 12 | diagnosis: `--no-sync` timing runs (equal/balanced, G=64/128) and repeated 1-epoch equal/balanced pairs |
| `configs/queue_single_A.txt` | 4 | single-GPU references on node A: G=128 seeds 1-3, G=64 seed 1, 30 epochs |
| `configs/queue_single_B.txt` | 3 | single-GPU references on node B: G=128 seeds 1-3, 30 epochs |
| `configs/queue_main.txt` | 8 | main 2-node runs: equal 64,64 vs balanced 75,53 at G=128 (seeds 1-3, 12 epochs), equal 128,128 vs balanced 136,120 at G=256 (seed 1, 16 epochs) |
| `configs/queue_r6.txt` | 8 | R6: node A slowed 2x and 3x (`--slow A:2/3`), equal 64,64 vs balanced 54,74 (F=2) / 41,87 (F=3), G=128, 1 epoch, 2 repeats |
| `configs/queue.txt` | 0 | empty template (comments only) |

## Folder map

| folder | contents |
|---|---|
| `scripts/` | `train.py` (training, DDP, checkpoints, logs), `run_queue.py` (queues), `bench_batch.py` (step time vs batch size), `make_split.py` (balanced split from two benchmark CSVs), `grad_check.py` (DDP gradient check), `net_test.py` (NCCL all_reduce test), `eval_report.py` (per-class test metrics of a best checkpoint), `env_check.py` (environment check) |
| `configs/` | `base.yaml` (defaults: global batch 64, epochs 5, seed 42, lr 1e-3) and the queue files |
| `results/` | committed results (see below), benchmark CSVs, smoke table |
| `logs/` | per-run logs on each node (not in git): `config_rank<r>.json`, `steps_rank<r>.csv`, `epochs_rank<r>.csv`, `test.json` |
| `checkpoints/` | per-run checkpoints on each node (not in git) |
| `reports/` | phase reports with technical output (link and NCCL tests) |
| `data/` | FI-2010 text files and `data/cache/` (not in git) |

## Results layout

| file | contents |
|---|---|
| `results/logs_A/` | node A copies of each run's logs: `epochs_rank0.csv`, `config_rank0.json`, `test.json` for the main, R6, single-GPU (`sA_*`) and kill-test runs; `steps_rank0.csv` for the main and R6 runs. `m_eq128_s1_bad10M` is the main run interrupted while the link was at 10 Mb/s (kept for reference, no `test.json`) |
| `results/logs_B/` | node B copies: rank 1 logs (`*_rank1.*`) of the main and R6 runs and the node B single-GPU runs (`sB_*`, rank 0 files) |
| `results/env_A.txt`, `results/env_B.txt` | environment of each node: `nvidia-smi`, Python, torch / CUDA / cuDNN / NCCL versions, OS, CPU, RAM, `pip freeze` |
| `results/main_summary.csv` | the 8 main runs: mode, G, split, seed, epochs, test accuracy, test macro-F1, best epoch, mean epoch time, samples/s, rank 0 median step time, `link_changed` |
| `results/r6_summary.csv` | the 8 R6 runs: slow factor, split, epoch time, samples/s, rank 0 and rank 1 median step time |
| `results/eval_m_bal128_s1.json` | test-set evaluation of the best checkpoint of `m_bal128_s1`: per-class precision, recall, F1 and support (up, stationary, down), macro and weighted averages, confusion matrix (counts and row-normalised) |

Times in the summaries: epoch time = slowest rank's training-loop time without validation and
checkpoint saves; median step times ignore the first 50 steps of each epoch.
