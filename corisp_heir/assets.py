"""Verify externally supplied assets; never substitute generated prototypes."""
import hashlib
from pathlib import Path

HEIR_PROTOTYPES_SHA256 = '6a2d47645ec47d25a95d05e04828de1f6cf9083a7f1a86f8e75d03949c387d2a'


def require_asset(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing required asset: {path.name}; expected sha256 {expected}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Asset checksum mismatch: {path.name}; expected sha256 {expected}')
    return path
