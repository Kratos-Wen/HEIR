"""Download the pinned HEIR paper dataset and verify SHA-256 checksums.

Uses the Python standard library and one download at a time. On clusters, run
full downloads and verification on a compute node with explicit resources.
"""
import argparse
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = 8


def destination(root, relative):
    """Reject paths that could write outside the selected dataset directory."""
    root = root.resolve()
    result = (root / relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(root) or result == root:
        raise ValueError(f"Unsafe dataset path: {relative}")
    return result


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream(url, output):
    """Retry connection failures, rate limits and server errors; other HTTP errors stop the download."""
    for attempt in range(1, ATTEMPTS + 1):
        wait = 2 ** attempt
        try:
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
            return
        except HTTPError as error:
            if attempt == ATTEMPTS or (error.code != 429 and error.code < 500):
                raise
            retry_after = error.headers.get("Retry-After", "") if error.headers else ""
            wait = max(wait, int(retry_after) if retry_after.isdigit() else 0)
        except (OSError, HTTPException):
            if attempt == ATTEMPTS:
                raise
        output.seek(0)
        output.truncate()
        time.sleep(wait)


def fetch(url, path, expected, size, verify_only=False):
    """Keep existing files intact; install a new download only after validation."""
    if path.exists():
        if path.stat().st_size != size or checksum(path) != expected:
            raise ValueError(f"Checksum/size mismatch in existing file: {path}")
        return
    if verify_only:
        raise FileNotFoundError(f"Missing dataset file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        # Exclusive creation prevents simultaneous runs from sharing a partial file.
        with temporary.open("xb") as output:
            try:
                stream(url, output)
            except Exception:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
    except FileExistsError:
        raise FileExistsError(f"Partial download already exists: {temporary}") from None
    try:
        if temporary.stat().st_size != size or checksum(temporary) != expected:
            raise ValueError(f"Downloaded checksum/size mismatch: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/HEIR")
    parser.add_argument("--annotations-only", action="store_true", help="skip image downloads")
    parser.add_argument("--verify-only", action="store_true", help="verify local files without network access")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/heir_dataset.json").read_text())
    for key in ("revision", "image_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get(key) or ""):
            parser.error(f"Dataset {key} is not a published commit")
    base = f"https://huggingface.co/datasets/{manifest['repo_id']}/resolve/"
    for relative, spec in manifest["files"].items():
        fetch(base + manifest["revision"] + "/" + quote(spec["path"], safe="/"),
              destination(args.output, relative), spec["sha256"], spec["size"], args.verify_only)
    for split, expected in manifest["split_counts"].items():
        annotation = json.loads((args.output / "annotations" / f"{split}.json").read_text())
        if annotation["split"] != split or len(annotation["images"]) != expected:
            raise ValueError(f"Incorrect split metadata: {split}")
    print("Verified annotations and vocabulary: train 15158, val 615, test 2957")
    if not args.annotations_only:
        images = json.loads((args.output / "image_files.json").read_text())
        if len(images) != manifest["image_count"]:
            raise ValueError("Incorrect image manifest length")
        for index, spec in enumerate(images, 1):
            fetch(base + manifest["image_revision"] + "/" + quote(spec["path"], safe="/"),
                  destination(args.output, spec["file_name"]), spec["sha256"], spec["size"], args.verify_only)
            if index % 500 == 0 or index == len(images):
                print(f"Verified images: {index}/{len(images)}", flush=True)
    print(f"HEIR data directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
