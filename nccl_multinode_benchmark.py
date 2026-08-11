#!/usr/bin/env python3
"""
NCCL all-reduce benchmark across a Ray cluster (2x H200 nodes, 8 GPUs each = 16 ranks).

Runs as the Ray job driver on the head node. It launches one Ray actor per GPU,
wires them into a single torch.distributed NCCL process group, and measures
all-reduce latency/bandwidth over a range of message sizes -- reporting algorithm
bandwidth (algBW) and bus bandwidth (busBW) the same way nccl-tests' all_reduce_perf does.

busBW = algBW * 2 * (n - 1) / n   for ring all-reduce over n ranks.

Entry point matches the RayJob:
    /opt/conda/envs/ray38/bin/python /home/ray/workspace/nccl_benchmark.py
"""

import os
import time
import socket
import datetime

import ray
import torch
import torch.distributed as dist

# ----------------------------------------------------------------------------
# Config (override via env vars if you like)
# ----------------------------------------------------------------------------
GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", "8"))
NUM_NODES     = int(os.environ.get("NUM_NODES", "2"))
WORLD_SIZE    = GPUS_PER_NODE * NUM_NODES           # 16

MASTER_PORT   = int(os.environ.get("MASTER_PORT", "29500"))
DTYPE         = torch.float32                        # 4 bytes/elem
WARMUP_ITERS  = int(os.environ.get("WARMUP_ITERS", "5"))
TIMED_ITERS   = int(os.environ.get("TIMED_ITERS", "20"))
INIT_TIMEOUT  = datetime.timedelta(minutes=30)

# Message sizes to sweep: 1 MiB .. 8 GiB, doubling. Trim the top end if you hit OOM.
MIN_BYTES = int(os.environ.get("MIN_BYTES", str(1 << 20)))   # 1 MiB
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(8 << 30)))   # 8 GiB


@ray.remote(num_gpus=1)
class NCCLWorker:
    def __init__(self, rank, world_size, master_port):
        self.rank = rank
        self.world_size = world_size
        self.master_port = master_port
        self.master_addr = None
        # Ray sets CUDA_VISIBLE_DEVICES so this actor's single GPU is always cuda:0.
        self.device = torch.device("cuda:0")

    def node_ip(self):
        return ray.util.get_node_ip_address()

    def setup(self, master_addr):
        self.master_addr = master_addr
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        # NCCL_* env vars come from the pod spec; leave them alone.

        torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
            timeout=INIT_TIMEOUT,
        )
        dist.barrier()
        return (self.rank, socket.gethostname(), torch.cuda.get_device_name(self.device))

    def _sync_all(self):
        torch.cuda.synchronize()
        dist.barrier()

    def _time_allreduce(self, tensor, iters):
        """Return average seconds per all-reduce, worst-case across all ranks."""
        self._sync_all()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end.record()
        torch.cuda.synchronize()

        local_ms = start.elapsed_time(end) / iters
        # Take the max across ranks so the reported time bounds the collective.
        t = torch.tensor([local_ms], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item() / 1000.0  # seconds

    def benchmark(self, sizes_bytes):
        results = []
        for nbytes in sizes_bytes:
            numel = max(1, nbytes // DTYPE.itemsize)
            actual_bytes = numel * DTYPE.itemsize
            tensor = torch.full((numel,), float(self.rank + 1),
                                dtype=DTYPE, device=self.device)

            # Correctness check FIRST, on a fresh tensor with a single all-reduce.
            # Each rank fills with (rank+1), so SUM over ranks 1..N == N*(N+1)/2.
            check_t = torch.full((numel,), float(self.rank + 1),
                                 dtype=DTYPE, device=self.device)
            dist.all_reduce(check_t, op=dist.ReduceOp.SUM)
            expected = self.world_size * (self.world_size + 1) / 2.0
            ok = bool(torch.isclose(check_t[0], torch.tensor(expected, device=self.device),
                                    rtol=1e-3).item())
            del check_t

            # Warmup (values may grow across in-place iters; that's fine, we only time it)
            self._time_allreduce(tensor, WARMUP_ITERS)

            # Timed run
            sec = self._time_allreduce(tensor, TIMED_ITERS)

            alg_bw = actual_bytes / sec / 1e9                       # GB/s
            bus_bw = alg_bw * 2 * (self.world_size - 1) / self.world_size
            results.append((actual_bytes, sec * 1e6, alg_bw, bus_bw, ok))
            del tensor
            torch.cuda.empty_cache()

        return self.rank, results

    def teardown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        return True


def human(nbytes):
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes/1:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.0f} GiB"


def fmt_bytes(nbytes):
    if nbytes >= (1 << 30):
        return f"{nbytes / (1 << 30):.2f} GiB"
    if nbytes >= (1 << 20):
        return f"{nbytes / (1 << 20):.2f} MiB"
    if nbytes >= (1 << 10):
        return f"{nbytes / (1 << 10):.2f} KiB"
    return f"{nbytes} B"


def main():
    ray.init(address="auto")

    print(f"Cluster resources: {ray.cluster_resources()}")
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < WORLD_SIZE:
        print(f"WARNING: cluster reports {total_gpus} GPUs but WORLD_SIZE={WORLD_SIZE}. "
              f"Actors may not all schedule.")

    # Build message-size sweep
    sizes = []
    n = MIN_BYTES
    while n <= MAX_BYTES:
        sizes.append(n)
        n *= 2

    # One actor per GPU. Ray spreads them across the two worker nodes.
    workers = [NCCLWorker.remote(rank, WORLD_SIZE, MASTER_PORT)
               for rank in range(WORLD_SIZE)]

    # Rank 0's node IP is the rendezvous master.
    master_addr = ray.get(workers[0].node_ip.remote())
    print(f"Rendezvous master: {master_addr}:{MASTER_PORT}")

    placement = ray.get([w.setup.remote(master_addr) for w in workers])
    print("Process group initialized. Rank placement:")
    for rank, host, gpu in sorted(placement):
        print(f"  rank {rank:2d}  {host}  {gpu}")

    # Run benchmark on all ranks; results are identical, so keep rank 0's.
    all_results = ray.get([w.benchmark.remote(sizes) for w in workers])
    rank0 = next(r for rk, r in all_results if rk == 0)

    print("\n" + "=" * 78)
    print(f"NCCL all-reduce  |  world_size={WORLD_SIZE}  "
          f"({NUM_NODES} nodes x {GPUS_PER_NODE} GPU)  |  dtype={DTYPE}")
    print("=" * 78)
    print(f"{'size':>12} {'time (us)':>12} {'algBW (GB/s)':>14} "
          f"{'busBW (GB/s)':>14} {'check':>7}")
    print("-" * 78)
    for nbytes, us, alg, bus, ok in rank0:
        print(f"{fmt_bytes(nbytes):>12} {us:>12.1f} {alg:>14.2f} "
              f"{bus:>14.2f} {'OK' if ok else 'FAIL':>7}")
    print("=" * 78)
    peak = max(rank0, key=lambda r: r[3])
    print(f"Peak busBW: {peak[3]:.2f} GB/s at {fmt_bytes(peak[0])}")

    ray.get([w.teardown.remote() for w in workers])
    print("Done.")


if __name__ == "__main__":
    main()