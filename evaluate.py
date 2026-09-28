#!/usr/bin/env python3
"""Evaluate a released HEIR model from one configuration file.

    python evaluate.py configs/heir/rlipv2_swinl.toml
    python evaluate.py --list

A model configuration names the model directory and variant, the released checkpoint and the metrics. Local paths
(dataset, support table, per-model Python environments, inference assets, checkpoint cache) are read from
configs/paths.toml; copy configs/paths.example.toml to create it. The evaluator fetches the pinned model source,
links the assets, obtains and verifies the checkpoint, exports test predictions and runs the scorers. Existing
outputs of finished stages are reused unless --force is given. Device selection is left to the caller
(for example CUDA_VISIBLE_DEVICES).
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parent
METRICS = {'role': 'score-role', 'hoi': 'score-hoi', 'sets-topk': 'score-sets-topk', 'sets-map': 'score-sets-map'}
RESULT_DIRS = {'role': 'role', 'hoi': 'hoi', 'sets-topk': 'sets_topk', 'sets-map': 'sets_map'}


def load(path):
    with Path(path).open('rb') as handle:
        return tomllib.load(handle)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def todo(value):
    return not value or str(value).startswith('TODO')


def checkpoint_file(key, registry, paths, dry_run):
    """Return the verified local checkpoint, downloading it from its published link if needed."""
    entry = registry[key]
    if entry.get('released') is False:  # trained locally with the recipe in models/
        target = Path(paths.get('checkpoint_dir', 'checkpoints')).expanduser()
        target = (target if target.is_absolute() else ROOT / target) / entry['file']
        if not target.is_file() and not dry_run:
            raise SystemExit(f'{key} is not released; train it with its recipe and place the final checkpoint at {target}.')
        return target
    if todo(entry.get('file')) or todo(entry.get('sha256')):
        raise SystemExit(f'Checkpoint {key} is not released yet (configs/checkpoints.toml).')
    target = Path(paths.get('checkpoint_dir', 'checkpoints')).expanduser()
    target = (target if target.is_absolute() else ROOT / target) / entry['file']
    if not target.is_file():
        if todo(entry.get('url')):
            raise SystemExit(f'No download link for {key} yet; place {entry["file"]} in {target.parent}.')
        if dry_run:
            print(f'would download {entry["url"]} -> {target}')
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if 'drive.google.com' in entry['url']:
            if shutil.which('gdown') is None:
                raise SystemExit(f'Install gdown or download {entry["url"]} to {target} manually.')
            subprocess.run(['gdown', '--fuzzy', entry['url'], '-O', str(target)], check=True)
        elif entry['url'].startswith('https://') and entry['url'].rsplit('/', 1)[-1] == entry['file']:
            print(f'downloading {entry["url"]}', flush=True)
            partial = target.with_suffix(target.suffix + '.part')
            urllib.request.urlretrieve(entry['url'], partial)
            partial.rename(target)
        else:
            raise SystemExit(f'Download {entry["file"]} from {entry["url"]} into {target.parent} (no direct link).')
    if not dry_run and digest(target) != entry['sha256']:
        raise SystemExit(f'SHA-256 mismatch for {target}; expected {entry["sha256"]}.')
    return target


def run(model_dir, args, python, dry_run):
    command = [sys.executable, 'baseline.py', *args]
    if python and args[0] in ('run', 'show'):
        command += ['--python', python]
    print('+', ' '.join(command), flush=True)
    if not dry_run or args[0] in ('show', 'check'):
        subprocess.run(command, cwd=model_dir, check=True)


def collect(output, metrics):
    results = {}
    for metric in metrics:
        summary = output / RESULT_DIRS[metric] / 'summary.json'
        if not summary.is_file():
            continue
        data = json.loads(summary.read_text())
        if metric.startswith('sets'):
            results[metric] = {'set_mAP': data['set_mAP']}
        else:
            results[metric] = {k: data[k] for k in ('mAP', 'rare_mAP', 'non_rare_mAP', 'unseen_mAP') if k in data}
    return results


def evaluate(config_path, paths, registry, output, force, dry_run):
    config = load(config_path)
    model_dir = ROOT / 'models' / config['model']
    recipe = json.loads((model_dir / 'baseline.json').read_text())
    variant = config['variant']
    commands = recipe['commands']
    output = Path(output or ROOT / 'runs' / variant).resolve()
    envs = paths.get('python', {})
    python = envs.get(config.get('python', ''), envs.get('default', sys.executable))
    scorer_python = envs.get('scorer', envs.get('default', sys.executable))

    assets = {'HEIR': paths['heir'], 'HEIR_SUPPORT': paths['heir_support']}
    for key in config.get('assets', []):
        if key not in paths.get('assets', {}):
            raise SystemExit(f'{config_path}: set assets.{key} in the paths file.')
        assets[key] = paths['assets'][key]
    with (model_dir / '.evaluate.lock').open('w') as lock:  # configurations of one model may run concurrently
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(model_dir, ['run', 'sources', '--python', sys.executable], None, dry_run)
        run(model_dir, ['link'] + [f'--set={k}={Path(v).expanduser()}' for k, v in assets.items()], None, dry_run)

    checkpoint = checkpoint_file(config['checkpoint'], registry, paths, dry_run)
    expected = next(t for t in commands[f'{variant}-export']['argv'] if t.startswith('{output}/train/'))
    link = output / expected.removeprefix('{output}/')
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(checkpoint)

    stages = [('prepare', output / 'heir_hico'), ('support', output / 'support_mask.npy'),
              ('labels', None), ('export', output / 'test_predictions.jsonl')]
    for stage, product in stages:
        name = f'{variant}-{stage}'
        if name not in commands:
            continue
        if product is not None and product.exists() and not force:
            print(f'= {stage}: reusing {product}')
            continue
        if force and product is not None and product.exists() and not dry_run:
            shutil.rmtree(product) if product.is_dir() else product.unlink()
        run(model_dir, ['run', name, '--output', str(output)], python, dry_run)
    metrics = config.get('metrics', list(METRICS))
    for metric in metrics:
        result = output / RESULT_DIRS[metric] / 'summary.json'
        if result.exists() and not force:
            print(f'= {metric}: reusing {result}')
            continue
        run(model_dir, ['run', f'{variant}-{METRICS[metric]}', '--output', str(output)], scorer_python, dry_run)
    if dry_run:
        return
    results = {'config': str(Path(config_path)), 'name': config.get('name', variant), 'checkpoint': config['checkpoint'],
               'checkpoint_sha256': digest(checkpoint),
               'predictions_sha256': digest(output / 'test_predictions.jsonl'), 'metrics': collect(output, metrics)}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def vcoco_checkout(config, paths, dry_run):
    """Check out the pinned V-COCO baseline release of this repository (reproductions/vcoco-baselines)."""
    target = ROOT / 'external/vcoco-baselines'
    url = paths.get('repository', config['repository'])
    if not (target / '.git').exists():
        print(f'+ git clone {url} ({config["branch"]} @ {config["commit"][:12]})', flush=True)
        if dry_run:
            return target
        subprocess.run(['git', 'clone', '--quiet', '--branch', config['branch'], url, str(target)], check=True)
    if not dry_run:
        known = subprocess.run(['git', '-C', str(target), 'cat-file', '-e', config['commit'] + '^{commit}'],
                               capture_output=True).returncode == 0
        if not known:
            subprocess.run(['git', '-C', str(target), 'fetch', '--quiet', 'origin', config['branch']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--quiet', config['commit']], check=True)
        head = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
        if head != config['commit']:
            raise SystemExit(f'{target} is at {head}, expected {config["commit"]}')
    return target


def evaluate_vcoco(config_path, config, paths, registry, output, force, dry_run):
    """Released-checkpoint inference and official V-COCO role AP."""
    checkout = vcoco_checkout(config, paths, dry_run)
    model_dir = checkout / 'models' / config['model']
    prefix = f'{config["variant"]}-' if config.get('variant') else ''
    output = Path(output or ROOT / 'runs' / f'vcoco_{config.get("variant") or config["model"]}').resolve()
    envs = paths.get('python', {})
    python = envs.get(config.get('python', ''), envs.get('default', sys.executable))
    checkpoint = checkpoint_file(config['checkpoint'], registry, paths, dry_run)
    assets = {'VCOCO': paths['vcoco'], config['checkpoint_asset']: checkpoint}
    for key, source in config.get('assets', {}).items():
        if source not in paths.get('assets', {}):
            raise SystemExit(f'{config_path}: set assets.{source} in the paths file.')
        assets[key] = paths['assets'][source]
    if dry_run:
        print(f'+ baseline.py run sources; link {", ".join(assets)}; run {prefix}infer; run {prefix}score')
        return
    with (model_dir / '.evaluate.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(model_dir, ['run', 'sources', '--python', sys.executable], None, dry_run)
        run(model_dir, ['link'] + [f'--set={k}={Path(v).expanduser()}' for k, v in assets.items()], None, dry_run)
    recipe = json.loads((model_dir / 'baseline.json').read_text())
    score = recipe['commands'][f'{prefix}score']['argv']
    cache = output / score[score.index('--cache') + 1].removeprefix('{output}/')
    if cache.exists() and not force:
        print(f'= infer: reusing {cache}')
    else:
        run(model_dir, ['run', f'{prefix}infer', '--output', str(output)], python, dry_run)
    run(model_dir, ['run', f'{prefix}score', '--output', str(output)], envs.get('scorer', python), dry_run)
    metrics = json.loads((output / 'official_metrics.json').read_text())['primary_comparable_metrics']
    results = {'config': str(Path(config_path)), 'name': config.get('name'), 'checkpoint': config['checkpoint'],
               'checkpoint_sha256': registry[config['checkpoint']]['sha256'], 'cache_sha256': digest(cache),
               'metrics': metrics}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


def listing(registry):
    rows = []
    for path in sorted((ROOT / 'configs').glob('*/*.toml')):
        config = load(path)
        entry = registry.get(config.get('checkpoint', ''), {})
        state = ('train locally' if entry.get('released') is False else 'TODO' if todo(entry.get('sha256'))
                 else 'link TODO' if todo(entry.get('url')) else 'released')
        rows.append(f'{str(path.relative_to(ROOT)):40s} {config.get("name", ""):28s} checkpoint: {state}')
    print('\n'.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', nargs='?', type=Path)
    parser.add_argument('--paths', type=Path, default=ROOT / 'configs/paths.toml')
    parser.add_argument('--checkpoints', type=Path, default=ROOT / 'configs/checkpoints.toml')
    parser.add_argument('--output', type=Path, help='run directory (default: runs/<variant>)')
    parser.add_argument('--force', action='store_true', help='recompute stages whose outputs exist')
    parser.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    parser.add_argument('--list', action='store_true', help='list the model configurations')
    a = parser.parse_args()
    registry = load(a.checkpoints)
    if a.list:
        return listing(registry)
    if a.config is None:
        parser.error('give a configuration file or --list')
    config = load(a.config)
    if config.get('benchmark') == 'vcoco' and config.get('family') != 'corisp':
        if not a.paths.is_file():
            raise SystemExit(f'{a.paths} not found; copy configs/paths.example.toml and fill in local paths.')
        return evaluate_vcoco(a.config, config, load(a.paths), registry, a.output, a.force, a.dry_run)
    if config.get('family') == 'corisp':
        raise SystemExit(f'{a.config}: CoRISP runs from the main branch; see the command in this file.')
    if not a.paths.is_file():
        raise SystemExit(f'{a.paths} not found; copy configs/paths.example.toml and fill in local paths.')
    evaluate(a.config, load(a.paths), registry, a.output, a.force, a.dry_run)


if __name__ == '__main__':
    main()
