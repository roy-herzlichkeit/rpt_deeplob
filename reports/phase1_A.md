# Phase 1 — Node A

## Part 1: env_check.py

```
hostname: herzlichkeit-ubuntu
python: 3.12.3
torch: 2.11.0+cu128
torch.version.cuda: 12.8
cuda.is_available: True
nccl: 2.28.9
gpu: NVIDIA GeForce RTX 4060 Laptop GPU
gpu_memory_gb: 7.60
matmul 20x 4096x4096 fp32 (sanity check only): 0.325 s total, 16.2 ms/iter
```

## Part 2: Ethernet link (rpt-eth, eno1)

```
$ nmcli con up rpt-eth
Connection successfully activated (D-Bus active path: /org/freedesktop/NetworkManager/ActiveConnection/18)
$ ip -br a show eno1
eno1             UP             192.168.10.1/24
$ cat /sys/class/net/eno1/speed
1000
$ ping -c 4 -W 2 192.168.10.2
4 packets transmitted, 4 received, 0% packet loss, time 3071ms
rtt min/avg/max/mdev = 0.149/0.190/0.254/0.040 ms
$ /etc/ufw/ufw.conf
ENABLED=no   (ufw inactive, no rule added)
$ iperf3 -s -1 -B 192.168.10.1 -p 5202   (5201 taken by iperf3.service)
Accepted connection from 192.168.10.2, port 41424
[  5]   0.00-10.02  sec   112 MBytes  94.1 Mbits/sec                  receiver
$ cat /sys/class/net/eno1/speed /sys/class/net/eno1/duplex   (after iperf3)
100
full
```

Note: link negotiated 100 Mb/s full duplex during the test (earlier reading: 1000).
