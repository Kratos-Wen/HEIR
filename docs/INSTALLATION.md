# Installation

The reference environment is Linux, Python 3.11, PyTorch 2.5.1 and torchvision 0.20.1. The CPU example, numerical tests and set scorers do not need model weights or a CUDA extension. Image inference and training require a CUDA GPU and the H-DETR extension.

## CPU example and tests

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m examples.event_sets
python -m pytest -q
python scripts/check_release.py
```

Run these commands in an allocated compute job on a shared cluster. Thread limits also apply to subprocesses created by the tests.

## GPU training and inference

Use a separate environment from the CPU-only installation. For CUDA 12.4:

```bash
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt -r requirements-test.txt
```

For CUDA 11.8 or 12.1, use the matching wheel index in the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v251). Keep torch and torchvision versions paired. Compiling H-DETR requires a local CUDA toolkit, a compatible C++ compiler and a visible GPU; a PyTorch CUDA wheel alone does not supply `nvcc`.

Fetch the pinned third-party sources (PViC with its submodules and SL-HOI) into `vendor/`:

```bash
python scripts/fetch_vendor.py
```

Build the extension inside the allocated GPU job:

```bash
export MAX_JOBS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
(cd vendor/pvic/h_detr/models/ops && python setup.py build_ext --inplace)
python -c 'from corisp_heir import environment; import MultiScaleDeformableAttention; print("H-DETR extension available")'
python -m pytest -q
```

Compilation may use more memory than the numerical tests. Set the job's memory and wall-time allocation explicitly. Increase `MAX_JOBS` only within the allocated CPU count.

## Local inputs

No command downloads annotations or weights. Configure the files listed in [Assets](ASSETS.md) and follow [Training and Evaluation](REPRODUCTION.md). A source checkout must retain its `configs/`, `src/` and integration directories and the fetched `vendor/`; copying only the model file is insufficient.

CI exercises the CPU example, source integrity and synthetic tests. It does not train a model or certify benchmark reproduction without the required external assets.
