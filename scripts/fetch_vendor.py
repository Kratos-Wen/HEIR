"""Fetch the pinned third-party sources into vendor/ and verify the files this implementation uses.

PViC is checked out with its pinned submodules (H-DETR, DETR, Pocket, V-COCO and HICO-DET utilities). SL-HOI
receives configs/vendor_patches/slhoi.patch, which adds the DINOv3 and CLIP license texts next to the code copied
from those projects. Every file listed in configs/vendor_sha256.json must then match its recorded checksum.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    subprocess.run(["git", *map(str, args)], check=True)


def fetch(name, spec):
    dest = ROOT / "vendor" / name
    if not (dest / ".git").exists():
        if dest.exists():
            raise FileExistsError(f"{dest} exists but is not a checkout; remove it and retry")
        git("clone", "--quiet", spec["url"], dest)
        git("-C", dest, "checkout", "--quiet", spec["commit"])
        if spec.get("submodules"):
            git("-C", dest, "submodule", "update", "--quiet", "--init", "--recursive")
        if spec.get("patch"):
            git("-C", dest, "apply", "--whitespace=nowarn", ROOT / spec["patch"])
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != spec["commit"]:
        raise RuntimeError(f"vendor/{name} is at {head}, expected {spec['commit']}")


def verify():
    manifest = json.loads((ROOT / "configs/vendor_sha256.json").read_text())
    bad = [path for path, expected in manifest.items()
           if not (ROOT / path).is_file() or hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected]
    if bad:
        raise RuntimeError(f"Third-party files differ from the pinned release: {bad[:8]}")
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="check an existing vendor/ without fetching")
    args = parser.parse_args()
    if not args.verify_only:
        for name, spec in json.loads((ROOT / "configs/vendor_sources.json").read_text()).items():
            fetch(name, spec)
    print(f"Verified {verify()} third-party files in vendor/")


if __name__ == "__main__":
    main()
