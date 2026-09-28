"""Resolve the bundled model and externally supplied pretrained assets."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT
HOST = ROOT / 'vendor/pvic'
SLHOI = ROOT / 'vendor/slhoi'
WEIGHTS = Path(os.environ.get('CORISP_WEIGHTS', ROOT / 'weights'))
for path in reversed([ROOT / 'src', ROOT / 'integrations', HOST / 'h_detr/models/ops',
                      HOST, HOST / 'pocket', SLHOI]):
    sys.path.insert(0, str(path))


def restore_execution(execution):
    backend = execution.get('dp_backend', 'autograd')
    if backend not in ('autograd', 'adjoint'):
        raise ValueError(f'Unknown DP backend: {backend}')
    os.environ['HEIR_CORISP_DP_BACKEND'] = backend
    for key, variable in (
        ('compiled_dp', 'HEIR_CORISP_COMPILE_DP'),
        ('metadata_cache', 'HEIR_CORISP_METADATA_CACHE'),
        ('selective_recomputation', 'HEIR_CORISP_SELECTIVE_RECOMPUTE'),
    ):
        value = execution.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f'Execution setting {key} must be boolean')
        os.environ[variable] = str(int(value))


@lru_cache(maxsize=1)
def verify_core():
    manifest = json.loads((ROOT / 'configs/source_sha256.json').read_text())
    if not manifest:
        raise ValueError('Empty source manifest')
    for relative, expected in manifest.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source differs from the release manifest: {relative}')
    return manifest
