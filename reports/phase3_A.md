# Phase 3 — Node A (rank 0): NCCL all_reduce

```
$ cat /sys/class/net/eno1/speed   (before)
1000
$ NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.10.1 --master_port=29500 scripts/net_test.py --backend nccl
NCCL INFO Bootstrap: Using eno1:192.168.10.1<0>
NCCL INFO NCCL version 2.28.9+cuda12.9
NCCL INFO NET/Socket : Using [0]eno1:192.168.10.1<0>
NCCL INFO Using network Socket
NCCL INFO comm 0x39f0d290 rank 0 nRanks 2 nNodes 2 localRanks 1 localRank 0 MNNVL 0
NCCL INFO ncclCommInitRankConfig comm 0x39f0d290 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 1000 commId 0xdbe102b122b00837 - Init COMPLETE
[rank 0 | herzlichkeit-ubuntu | nccl] world_size=2 gpu=NVIDIA GeForce RTX 4060 Laptop GPU master=192.168.10.1:29500
NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0
NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Socket/0
NCCL INFO Connected all rings, use ring PXN 0 GDR 0
[rank 0 | herzlichkeit-ubuntu | nccl] test1 all_reduce SUM: got 3.0, expected 3.0 -> PASS
[rank 0 | herzlichkeit-ubuntu | nccl] test2 0.6 MB x200: median 5.312 ms, p95 5.519 ms
[rank 0 | herzlichkeit-ubuntu | nccl] test3 100 MB x10: median 853.5 ms, 117.2 MB/s (algbw)
exit=0
$ cat /sys/class/net/eno1/speed   (after)
1000
$ journalctl -k --since 22:54 | grep -iE 'eno1|Downshift'
(no output)
```
