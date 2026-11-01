"""Put the repository and third-party sources on the import path."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PVIC = ROOT / 'vendor/pvic'
SLHOI = ROOT / 'vendor/slhoi'
WEIGHTS = Path(os.environ.get('CORISP_WEIGHTS', ROOT / 'weights'))
for path in reversed([ROOT / 'src', ROOT / 'integrations', PVIC / 'h_detr/models/ops',
                      PVIC, PVIC / 'pocket', SLHOI]):
    sys.path.insert(0, str(path))


def restore_execution(execution):
    os.environ['HEIR_CORISP_COMPILE_DP'] = str(int(execution.get('compiled_dp', False)))
