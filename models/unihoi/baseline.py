#!/usr/bin/env python3
"""Run a baseline using explicit local assets and model configuration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe bundle destination: {relative}')
    return ROOT / path


def verify():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    bad = [p for p, sha in manifest.items()
           if not inside(p).is_file() or digest(inside(p)) != sha]
    if bad:
        raise RuntimeError(f'Source integrity mismatch: {bad}')
    print(f'Verified {len(manifest)} source files.')


def link_assets(recipe, assignments):
    assets = recipe['assets']
    pending = []
    for assignment in assignments:
        key, value = assignment.split('=', 1)
        asset = assets[key]
        source = Path(value).expanduser().resolve(strict=True)
        for destination, suffix in asset['mounts']:
            target = source / suffix if suffix else source
            if not target.exists():
                raise FileNotFoundError(target)
            dest = inside(destination)
            if dest.is_symlink() and dest.resolve() == target.resolve():
                continue
            if dest.exists() or dest.is_symlink():
                raise FileExistsError(f'Refusing to replace {dest}')
            pending.append((dest, target))
    for dest, target in pending:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(target, target_is_directory=target.is_dir())
        print(f'{dest.relative_to(ROOT)} -> {target}')


def check_assets(recipe, keys, hash_weights=False):
    for key in keys:
        for dest, _ in recipe['assets'][key]['mounts']:
            if not inside(dest).exists():
                raise FileNotFoundError(f'{key}: {dest}; see assets.example.json')
        for relative, expected in recipe['assets'][key].get('sha256', {}).items():
            if hash_weights and digest(inside(relative)) != expected:
                raise ValueError(f'Asset hash mismatch: {relative}')
    if 'VCOCO' in keys:
        data = ROOT / 'data/v-coco/data/splits'
        train = [int(x) for x in (data / 'vcoco_trainval.ids').read_text().split()]
        test = [int(x) for x in (data / 'vcoco_test.ids').read_text().split()]
        if len(train) != len(set(train)) or len(train) != 5400:
            raise ValueError('Expected 5,400 distinct official trainval IDs')
        if len(test) != len(set(test)) or len(test) != 4946 or set(train) & set(test):
            raise ValueError('Expected 4,946 disjoint official test IDs')
    print('Required paths and split IDs checked; images/checkpoints are not downloaded.')


def command_for(recipe, name, python, output, nnodes, nproc, node_rank, master_addr, port):
    spec = recipe['commands'][name]
    values = {'root': str(ROOT), 'python': python, 'output': str(output)}
    command = [token.format(**values) for token in spec['argv']]
    world = spec.get('world_size', 1)
    if spec.get('distributed'):
        nproc = nproc or world // nnodes
        if nproc * nnodes != world:
            raise ValueError(f'{name} requires world size {world}, not {nproc * nnodes}')
        if node_rank < 0 or node_rank >= nnodes:
            raise ValueError('Invalid node rank')
        launch = [python, '-m', 'torch.distributed.run', f'--nnodes={nnodes}',
                  f'--nproc-per-node={nproc}']
        if nnodes == 1:
            launch += ['--standalone']
        else:
            if not master_addr:
                raise ValueError('Multi-node launch requires --master-addr')
            launch += [f'--node-rank={node_rank}', f'--master-addr={master_addr}',
                       f'--master-port={port}']
        command = launch + command[1:]
    cwd = Path(spec['cwd'].format(**values))
    env = os.environ.copy()
    env.update({'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONHASHSEED': '42', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'WANDB_MODE': 'disabled'})
    env['PYTHONPATH'] = os.pathsep.join(str(inside(p)) for p in spec.get('pythonpath',
        ['workspace', spec['cwd'].replace('{root}/', '')]))
    return command, cwd, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'link', 'check', 'commands', 'run', 'show'])
    parser.add_argument('name', nargs='?')
    parser.add_argument('--set', action='append', default=[], dest='assignments')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--nproc-per-node', type=int)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--master-addr')
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--hash-assets', action='store_true')
    args = parser.parse_args()
    recipe = json.loads((ROOT / 'baseline.json').read_text())
    if args.action == 'verify':
        verify()
    elif args.action == 'link':
        link_assets(recipe, args.assignments)
    elif args.action == 'check':
        keys = recipe['commands'][args.name]['assets'] if args.name else recipe['assets']
        check_assets(recipe, keys, args.hash_assets)
    elif args.action == 'commands':
        print(json.dumps(recipe['commands'], indent=2))
    else:
        if args.name not in recipe['commands']:
            parser.error('Choose a command listed by `commands`')
        output = (args.output or ROOT / 'runs' / recipe['commands'][args.name].get('run_name', args.name)).resolve()
        command, cwd, env = command_for(recipe, args.name, args.python, output,
            args.nnodes, args.nproc_per_node, args.node_rank, args.master_addr, args.port)
        print(f'cd {shlex.quote(str(cwd))}\n{shlex.join(command)}', flush=True)
        if args.action == 'run':
            verify()
            check_assets(recipe, recipe['commands'][args.name]['assets'], args.hash_assets)
            output.mkdir(parents=True, exist_ok=True)
            # No shell expansion or scheduler. The caller owns device allocation.
            raise SystemExit(subprocess.call(command, cwd=cwd, env=env))


if __name__ == '__main__':
    main()
