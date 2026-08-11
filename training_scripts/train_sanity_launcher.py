#!/usr/bin/env python
"""
Ray launcher for the distributed sanity check.

Same Ray node-discovery + per-node torchrun placement as the Megatron launcher,
but it runs dist_sanity_train.py (zero external deps) instead of a Megatron
recipe. Use this to prove the 2-node x 8-GPU infra works before handing off to DS.
"""
import os
import shlex
import subprocess
import sys
 
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
 
 
def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: required env var {name!r} is not set. "
              f"Set it in the RayJob YAML's head container env.", file=sys.stderr)
        sys.exit(2)
    return val
 
 
NUM_NODES = int(_require_env("NUM_NODES"))
GPUS_PER_NODE = int(_require_env("GPUS_PER_NODE"))
TRAINING_SCRIPT = _require_env("TRAINING_SCRIPT")
TRAINING_ARGS = shlex.split(os.environ.get("TRAINING_ARGS", ""))
MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
 


@ray.remote(num_gpus=GPUS_PER_NODE)
def launch_on_node(node_rank: int, master_addr: str) -> int:
    env = os.environ.copy()
    
    # Force GPU visibility for torchrun and all spawned subprocesses
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(GPUS_PER_NODE))
    
    env.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    env.setdefault("NCCL_DEBUG", "INFO")
    
    cmd = [
        "python", "-m", "torch.distributed.run",
        f"--nnodes={NUM_NODES}",
        f"--nproc_per_node={GPUS_PER_NODE}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={MASTER_PORT}",
        TRAINING_SCRIPT,
        *TRAINING_ARGS,
    ]
    print(f"[node_rank={node_rank}] launching: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def main():
    print(f"config: NUM_NODES={NUM_NODES} GPUS_PER_NODE={GPUS_PER_NODE} "
          f"TRAINING_SCRIPT={TRAINING_SCRIPT} TRAINING_ARGS={TRAINING_ARGS}", flush=True)
    ray.init(address="auto")
    gpu_nodes = [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) >= GPUS_PER_NODE
    ]
    if len(gpu_nodes) < NUM_NODES:
        raise RuntimeError(
            f"Need {NUM_NODES} GPU nodes with >={GPUS_PER_NODE} GPUs, "
            f"found {len(gpu_nodes)}"
        )
    gpu_nodes = gpu_nodes[:NUM_NODES]
    master_addr = gpu_nodes[0]["NodeManagerAddress"]
    print(f"master_addr={master_addr}; nodes={[n['NodeManagerAddress'] for n in gpu_nodes]}", flush=True)

    futures = [
        launch_on_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote(node_rank=rank, master_addr=master_addr)
        for rank, node in enumerate(gpu_nodes)
    ]
    codes = ray.get(futures)
    print(f"return codes: {codes}", flush=True)
    if any(c != 0 for c in codes):
        sys.exit(1)
    print("ALL NODES SUCCEEDED — distributed infra verified.", flush=True)


if __name__ == "__main__":
    main()