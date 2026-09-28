import hashlib

import pytest

from corisp_heir.assets import require_asset


def test_missing_asset_never_regenerated(tmp_path):
    path = tmp_path / 'prototypes.pt'
    with pytest.raises(FileNotFoundError, match='expected sha256'):
        require_asset(path, '0' * 64)
    assert not path.exists()


def test_mismatched_asset_rejected(tmp_path):
    path = tmp_path / 'asset'
    path.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='checksum mismatch'):
        require_asset(path, '0' * 64)
    assert require_asset(path, hashlib.sha256(b'fixture').hexdigest()) == path
