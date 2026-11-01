from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arguments', [
    ['-m', 'corisp_heir.train'], ['-m', 'corisp_heir.predict'], ['-m', 'evaluation.native'], ['-m', 'evaluation.heir_sets'],
    ['scripts/evaluate_vcoco_sets.py'], ['scripts/evaluate_vcoco_official.py'],
])
def test_help_without_data_or_weights(arguments):
    result = subprocess.run([sys.executable, *arguments, '--help'], cwd=ROOT,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()
