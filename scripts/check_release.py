"""Check source manifests, documentation links and source-release hygiene."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".circleci", "__pycache__", ".pytest_cache", ".venv", ".idea",
             ".vscode", ".ruff_cache", "build", "dist"}
SKIP_ROOT_DIRS = {"weights", "outputs", "logs", "checkpoints", "data", "wandb", "htmlcov", "vendor"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".pth", ".pt", ".pkl", ".pickle",
                 ".jsonl", ".log", ".sqlite", ".db", ".zip", ".npz", ".npy", ".safetensors"}


def source_files(root):
    root = root.resolve()
    for directory, dirs, files in os.walk(root):
        base = Path(directory)
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS
                         and not (base == root and name in SKIP_ROOT_DIRS)
                         and not name.endswith(".egg-info")
                         and (not name.startswith(".") or (base == root and name == ".github")))
        for name in dirs:
            if (base / name).is_symlink():
                raise ValueError(f"Source symlink: {(base / name).relative_to(root)}")
        for name in sorted(files):
            path = base / name
            if (name in SKIP_DIRS or name in {"paths.local.sh", ".env", ".DS_Store"} or name.startswith(".env.")
                    or path.suffix in SKIP_SUFFIXES or ".local." in name):
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlink: {path.relative_to(root)}")
            yield path.relative_to(root), path


def manifests(root):
    # Third-party code is fetched into vendor/ by scripts/fetch_vendor.py and checked against
    # configs/vendor_sha256.json there; it is not part of the source release.
    source = {}
    for relative, path in source_files(root):
        if relative.parts[0] != "configs" and relative.suffix in {".py", ".sh"}:
            source[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source_sha256.json": source}


def check_manifests(root, update=False):
    for name, actual in manifests(root).items():
        path = root / "configs" / name
        if update:
            path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        recorded = json.loads(path.read_text())
        if recorded != actual:
            changed = sorted(key for key in set(recorded) | set(actual)
                             if recorded.get(key) != actual.get(key))
            raise ValueError(f"{name}: source mismatch in {', '.join(changed[:8])}")


def check_links(root):
    root = root.resolve()
    for relative, path in source_files(root):
        if path.suffix != ".md" or relative.parts[0] == "vendor":
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text()):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root) or not resolved.exists():
                raise ValueError(f"{relative}: broken local link {target}")


def check_hygiene(root):
    patterns = {
        "private absolute path": r"/(?:home|Users|scratch|mnt)/",
        "access token": r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}",
        "private key": r"-{5}BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY",
    }
    count = 0
    for relative, path in source_files(root):
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"Unexpected large source file: {relative}")
        content = path.read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            if re.search(pattern, relative.as_posix() + "\n" + content):
                raise ValueError(f"{label}: {relative}")
        if path.suffix == ".py":
            compile(content, str(relative), "exec")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-manifests", action="store_true",
                        help="Refresh checksums after intentional, reviewed source changes")
    args = parser.parse_args()
    try:
        count = check_hygiene(ROOT)
        check_links(ROOT)
        check_manifests(ROOT, update=args.update_manifests)
    except (ValueError, OSError, SyntaxError) as error:
        parser.exit(1, f"Release check failed: {error}\n")
    print(f"Release checks passed: {count} source files; manifests, local links and hygiene")


if __name__ == "__main__":
    main()
