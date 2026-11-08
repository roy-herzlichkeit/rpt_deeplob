"""Environment check for a DeepLOB DDP node."""
import platform
import socket
import time

import torch


def nccl_version():
    try:
        v = torch.cuda.nccl.version()
        return ".".join(map(str, v)) if isinstance(v, tuple) else str(v)
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"


def matmul_time(n=4096, warmup=3, iters=20):
    dev = torch.device("cuda")
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print(f"hostname: {socket.gethostname()}")
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    avail = torch.cuda.is_available()
    print(f"cuda.is_available: {avail}")
    print(f"nccl: {nccl_version()}")
    if not avail:
        print("gpu: none")
        return
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}")
    print(f"gpu_memory_gb: {p.total_memory / 1024**3:.2f}")
    t = matmul_time()
    print(f"matmul 20x 4096x4096 fp32 (sanity check only): {t:.3f} s total, "
          f"{t / 20 * 1000:.1f} ms/iter")


if __name__ == "__main__":
    main()
