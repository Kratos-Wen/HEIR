#!/usr/bin/env python3
"""Write a release copy of a training checkpoint: model weights, epoch and path-free arguments.

Optimizer, scheduler and other training state are dropped. Every absolute path stored in the saved arguments is
replaced by its final component, so no local directory names are published; the export commands pass their data,
text-encoder and CLIP paths explicitly. --set KEY=VALUE stores a release-relative value for an argument the export
reads from the checkpoint (e.g. clip_model=weights/ViT-B-32.pt). The output is refused if it already exists.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import torch


def clean(value):
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    if isinstance(value, (list, tuple)):
        return type(value)(clean(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    checkpoint = torch.load(a.input, map_location='cpu', weights_only=False)
    release = {'model': checkpoint['model']}
    if 'epoch' in checkpoint:
        release['epoch'] = checkpoint['epoch']
    if 'args' in checkpoint:
        args = copy.deepcopy(checkpoint['args'])
        for key, value in vars(args).items():
            setattr(args, key, clean(value))
        for item in a.set:
            key, value = item.split('=', 1)
            if not hasattr(args, key):
                raise KeyError(key)
            setattr(args, key, value)
        release['args'] = args
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, a.output)
    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    print(f'{a.output.name} {a.output.stat().st_size} bytes sha256 {digest}; kept {sorted(release)}')


if __name__ == '__main__':
    main()
