#!/usr/bin/env python3
"""Clone the pinned upstream sources listed in sources.json and apply this directory's adapters.

Each entry names a repository and commit, optionally initialises its pinned submodules and applies a patch from
patches/.
"""

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    subprocess.run(['git', *map(str, args)], check=True)


def merge(source, target):
    """Move links and files made before the checkout into it, keeping upstream files untouched."""
    for child in sorted(source.iterdir()):
        destination = target / child.name
        if child.is_dir() and not child.is_symlink() and destination.is_dir():
            merge(child, destination)
        elif destination.exists() or destination.is_symlink():
            raise FileExistsError(f'{destination} exists in the upstream checkout')
        else:
            child.rename(destination)
    source.rmdir()


def main():
    sources = json.loads((ROOT / 'sources.json').read_text())
    for name, spec in sources.items():
        dest = ROOT / 'workspace/reference_repos' / name
        if not (dest / '.git').exists():
            clone = dest.with_name(dest.name + '.clone')
            git('clone', '--quiet', spec['url'], clone)
            git('-C', clone, 'checkout', '--quiet', spec['commit'])
            if spec.get('submodules'):
                git('-C', clone, 'submodule', 'update', '--quiet', '--init', '--recursive')
            if spec.get('patch'):
                git('-C', clone, 'apply', '--whitespace=nowarn', ROOT / spec['patch'])
            if dest.exists():  # asset links created before the checkout
                merge(dest, clone)
            clone.rename(dest)
        head = subprocess.run(['git', '-C', str(dest), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
        if head != spec['commit']:
            raise RuntimeError(f'{name}: checked out {head}, expected {spec["commit"]}')
        print(f'{name}: {spec["url"]} @ {spec["commit"][:12]}')


if __name__ == '__main__':
    main()
