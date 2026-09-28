# Contributing

Keep changes focused and include a test for the behavior being changed. Use [Installation](docs/INSTALLATION.md) to set up the CPU test environment before modifying the implementation.

## Development checks

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/check_release.py --update-manifests
python -m pytest -q
python -m examples.event_sets
python scripts/check_release.py
git diff --check
```

Update manifests only after an intentional source change and review their diff. Training and prediction check these manifests to detect source changes. Refreshing them does not establish numerical equivalence to the release; the tests and evaluation protocol still apply.

## Changes to model or evaluation behavior

Document affected tensor shapes, assignment support, score normalization and matching rules. Changes to the exact partition should agree with exhaustive enumeration on small cases, including gradients and saturated role counts. Changes to the set scorer should test missing and additional members, shared identities and tied scores. Preserve the published metric definitions unless a change is explicitly identified as a new protocol.

Do not commit third-party source. Change a pinned dependency through `configs/vendor_sources.json`, `configs/vendor_patches/` and `configs/vendor_sha256.json`, separately from model changes.

## Pull requests and issues

Describe the problem, the proposed change, and the commands used to test it. Use a small synthetic reproduction when possible. Include no datasets, images, checkpoints, prediction caches, machine-specific paths or credentials. The issue and pull-request templates list the information needed to reproduce a problem.

Contributions must be yours to submit. Original contributions use the repository's [noncommercial license](LICENSE); third-party portions retain their own terms. Keep discussions technical and respectful. For vulnerabilities, follow [Security](SECURITY.md) rather than opening a public issue with exploit details.
