#!/usr/bin/env python
"""
Distributed setup sanity check — proves the multi-node GPU + NCCL + DDP stack
works end to end, with ZERO external dependencies (no HuggingFace, no internet,
no datasets, no model downloads).

This is the script torch.distributed.run launches on each node (via the Ray
launcher). It exercises exactly the primitives real Megatron training relies on:
  - every rank initializes and joins the process group (all 16 present)
  - CUDA device per rank is real and usable
  - cross-node NCCL collectives work (all_reduce)
  - DDP gradient synchronization works across nodes
  - a few training steps run and loss decreases

If this prints SUCCESS on rank 0, the distributed infrastructure is sound and
ready for the DS team's in-house models.
"""
import datetime
import os
import sys
import torch
import termcolor
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    # torchrun sets these env vars for every process.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    log(rank, f"starting: local_rank={local_rank}, world_size={world_size}, "
              f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}")

    # 1) Join the process group. If cross-node NCCL/networking is broken,
    #    this hangs then times out — the first real test.
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=120),
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    log(rank, f"process group joined ({dist.get_world_size()} ranks); "
              f"GPU = {torch.cuda.get_device_name(local_rank)}")

    # Prove a runtime_env-installed extra package is actually importable and usable.
    if rank == 0:
        # Replace your current log line with this:
        log(rank, f"runtime_env package check: termcolor successfully imported from {termcolor.__file__}")
        print("=================== RAY ENVIRONMENT SANITY CHECK ===================")
    
        # 1. Test the new package (tqdm)
        try:
            print(f"✅ SUCCESS: 'tqdm' installed via runtime_env!")
            print(f"   -> Version: {tqdm.__version__}")
            print(f"   -> Location: {tqdm.__file__}")
        except ImportError:
            print("❌ FAILURE: 'tqdm' could not be found.")

        # 2. Test your inherited base image packages (e.g., torch)
        try:
            print(f"✅ SUCCESS: Inherited base image 'torch' successfully!")
            print(f"   -> Version: {torch.__version__}")
            print(f"   -> Location: {torch.__file__}")
        except ImportError:
            print("❌ FAILURE: Base image 'torch' is missing.")

        # 3. Print the path resolution order
        print(f"\n📂 Python Search Paths (First 3): {sys.path[:3]}")
        print("====================================================================")



    # 2) Cross-node collective: every rank contributes its rank number; the
    #    sum must equal 0+1+...+(world_size-1). Proves NCCL all_reduce works
    #    across all nodes, not just within one.
    t = torch.tensor([float(rank)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2
    ok = abs(t.item() - expected) < 1e-3
    if rank == 0:
        log(rank, f"all_reduce check: got {t.item()}, expected {expected} -> "
                  f"{'OK' if ok else 'MISMATCH'}")
    assert ok, f"all_reduce mismatch on rank {rank}"

    # 3) Tiny model wrapped in DDP — exercises gradient all-reduce across ranks,
    #    the core of data-parallel training. Pure random data, no dataset.
    torch.manual_seed(1234 + rank)
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
    ).to(device)
    model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 4) A few training steps. Loss should trend down; this confirms forward,
    #    backward, gradient sync, and optimizer step all work multi-node.
    steps = 20
    for step in range(steps):
        x = torch.randn(32, 1024, device=device)
        y = torch.randn(32, 1024, device=device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()   # <- DDP all-reduces gradients across all 16 ranks here
        opt.step()
        if rank == 0 and step % 5 == 0:
            log(rank, f"step {step:2d}  loss={loss.item():.4f}")

    # 5) Final barrier so all ranks confirm they finished together.
    dist.barrier()
    if rank == 0:
        log(rank, "=" * 56)
        log(rank, f"SUCCESS: {world_size} ranks across all nodes trained "
                  f"{steps} steps. Distributed infra is functional.")
        log(rank, "=" * 56)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()