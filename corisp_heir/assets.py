"""Verify externally supplied assets; never substitute generated prototypes."""
import hashlib
from pathlib import Path

HEIR_PROTOTYPES_SHA256 = '28cc193618258d1c8c4d2b38af562a6c7bdbd4001979fc092dd96e3f04bca522'


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
