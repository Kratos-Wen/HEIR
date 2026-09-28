"""Reject heavyweight evaluation outside an allocated Slurm compute step."""
import os
import socket


def require_compute_step(workers=1):
    node = os.environ.get("SLURMD_NODENAME", "").split(".")[0]
    local = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or not node or node != local:
        raise SystemExit("Refusing full evaluation on a login node. Use sbatch or an allocated srun compute step.")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if cpus < workers or workers < 1:
        raise SystemExit("Explicit SLURM_CPUS_PER_TASK must cover all evaluation workers.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
