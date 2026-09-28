"""Integrity and filesystem guarantees for dataset downloads; no network needed."""
import hashlib
import io
from pathlib import Path

import pytest

from scripts.download_heir import destination, fetch


def test_dataset_paths_cannot_escape_output(tmp_path):
    assert destination(tmp_path, "annotations/train.json") == tmp_path / "annotations/train.json"
    for path in ("../escape", str(tmp_path.parent / "outside"), "."):
        with pytest.raises(ValueError, match="Unsafe dataset path"):
            destination(tmp_path, path)
    (tmp_path / "external").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        destination(tmp_path, "external/escape")


def test_bad_download_is_never_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.download_heir.urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    path = tmp_path / "annotation.json"
    with pytest.raises(ValueError, match="Downloaded checksum"):
        fetch("https://example.invalid/file", path, hashlib.sha256(b"good").hexdigest(), 4)
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_existing_data_is_preserved_and_valid_data_needs_no_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected network request")
    monkeypatch.setattr("scripts.download_heir.urlopen", no_network)
    path = tmp_path / "annotation.json"
    path.write_bytes(b"good")
    digest = hashlib.sha256(b"good").hexdigest()
    fetch("https://example.invalid/file", path, digest, 4, verify_only=True)
    with pytest.raises(ValueError, match="existing file"):
        fetch("https://example.invalid/file", path, "0" * 64, 4)
    assert path.read_bytes() == b"good"
    with pytest.raises(FileNotFoundError, match="Missing dataset file"):
        fetch("https://example.invalid/file", tmp_path / "missing", digest, 4, verify_only=True)


def test_successful_download_and_stale_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.download_heir.urlopen", lambda *a, **k: io.BytesIO(b"good"))
    digest = hashlib.sha256(b"good").hexdigest()
    path = tmp_path / "annotation.json"
    partial = tmp_path / "annotation.json.part"
    partial.write_bytes(b"in progress")
    with pytest.raises(FileExistsError, match="Partial download"):
        fetch("https://example.invalid/file", path, digest, 4)
    assert partial.read_bytes() == b"in progress"
    partial.unlink()
    fetch("https://example.invalid/file", path, digest, 4)
    assert path.read_bytes() == b"good"
    assert not partial.exists()
