#!/usr/bin/env python
"""
Submit a multi-node distributed training job to the shared H200 Ray cluster
via the Job Submission API -- no kubectl / cluster access required.

Script distribution model: gcsfuse, NOT working_dir. Your training script and
train_launcher_generic.py must already be uploaded to GCS and visible under
/shared/... on every node (head and workers), via the gcsfuse CSI mount
already configured on the cluster -- exactly as validated in earlier runs.
This script does NOT ship any local files to the cluster.

Usage:
    python submit_job.py

Before running:
  1. Upload your training script to GCS so it lands under /shared/... :
       gcloud storage cp my_train.py gs://<bucket>/<path>/my_train.py
     (train_launcher_generic.py is platform-provided and should already be
     uploaded once by the platform team -- confirm its /shared/... path below.)
  2. Fill in RAY_ADDRESS with the exposed dashboard FQDN from your platform team.
  3. Set TRAINING_SCRIPT to the ABSOLUTE /shared/... path (matches the gcsfuse
     mount path on the cluster, NOT a local path).
  4. Adjust NUM_NODES / GPUS_PER_NODE / TRAINING_ARGS for your job.
  5. If you need extra pure-Python packages (no torch/CUDA dependency), list
     them under EXTRA_PIP_PACKAGES -- see the SOP for the limitations of this.
"""
from ray.job_submission import JobSubmissionClient

import os
os.environ.pop("RAY_ADDRESS", None) 


LAUNCHER_SCRIPT = "/shared/data_access/harsha/train_sanity_launcher.py"  # platform-provided


##### Fill the values here

RAY_ADDRESS = ""
NUM_NODES = "2"
GPUS_PER_NODE = "8"
TRAINING_SCRIPT = "/shared/data_access/harsha/dist_sanity_train.py"   # <-- your uploaded script
TRAINING_ARGS = ""                      

# ---------------------------------------------------------------------------
# 3. Optional: extra pure-Python packages (no torch/CUDA). Leave empty list
#    if not needed. See SOP section 4a for why this can't include torch-
#    dependent packages -- use a custom image for those instead.
# ---------------------------------------------------------------------------
EXTRA_PIP_PACKAGES = ["termcolor==2.4.0", "tqdm"]   # e.g. ["pandas==2.1.0", "tqdm"]

# ---------------------------------------------------------------------------
# 4. A meaningful, unique submission ID -- STRONGLY recommended on a shared
#    cluster so your job is identifiable on the dashboard among everyone
#    else's. Include your name/team and something job-specific.
# ---------------------------------------------------------------------------
SUBMISSION_ID = "harsha-fkecomm-test1"   # <-- change this every run
METADATA = {"user": "harsha.agrawal", "team": "mlp"}


#########################


def main():
    client = JobSubmissionClient(RAY_ADDRESS)

    runtime_env = {
        # NOTE: no working_dir -- nothing local is shipped. The launcher and
        # training script are both already on the gcsfuse /shared mount.
        "env_vars": {
            "NUM_NODES": NUM_NODES,
            "GPUS_PER_NODE": GPUS_PER_NODE,
            "TRAINING_SCRIPT": TRAINING_SCRIPT,
            "TRAINING_ARGS": TRAINING_ARGS,
        },
    }

    if EXTRA_PIP_PACKAGES:
        runtime_env["pip"] = {
            "packages": EXTRA_PIP_PACKAGES,
            "pip_check": False,
        }

    job_id = client.submit_job(
        entrypoint=f"python {LAUNCHER_SCRIPT}",
        submission_id=SUBMISSION_ID,
        metadata=METADATA,
        runtime_env=runtime_env,
    )

    print(f"Submitted job: {job_id}")
    print(f"Tail logs with:")
    print(f"  ray job logs {job_id} --address {RAY_ADDRESS} --follow")
    print(f"Check status with:")
    print(f"  ray job status {job_id} --address {RAY_ADDRESS}")
    print(f"Stop it with:")
    print(f"  ray job stop {job_id} --address {RAY_ADDRESS}")


if __name__ == "__main__":
    main()