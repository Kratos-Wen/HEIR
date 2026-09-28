"""Separate, sealed closed-loop repair experiment using verified train caches.

The previous run, checkpoint, export and test metrics are never overwritten.
Training starts from a fresh seed42 DiT; qualification audits the failed model.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .closed_loop import trajectory, backward_rollout
from .detector import ROOT, digest, build_detector
from .diffusion import ForwardDiffusion, ground_truth, initialize
from .model import SliceDiT
from .run import training_batch, write_json, save_checkpoint, restore_rng, evaluate
from .vcoco import annotations, DATA


def source_hashes():
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += [ROOT / 'reference_repos/DiT/models.py',
                ROOT / 'vcoco_eval/scripts/train_idiff_closed_loop_existing.sh']
    return {str(p): digest(p) for p in sorted(sources)}


def load_cache(source, image_id):
    path = source / 'train_features' / f'{image_id:012d}.pth'
    meta = json.loads(path.with_suffix('.json').read_text())
    if digest(path) != meta['sha256'] or meta['image_id'] != image_id:
        raise ValueError('Original feature cache integrity mismatch')
    cache = torch.load(path, weights_only=True)
    if cache['image_id'] != image_id:
        raise ValueError('Wrong cache image ID')
    return cache


def prepare_batch(cache, epoch, device, process):
    values = training_batch(cache, epoch, device, 29)
    appearance, noun_prior, labels, known, nouns = [x.repeat_interleave(10, 0) for x in values]
    prior = initialize(noun_prior, 29)
    clean = ground_truth(nouns, labels, 81)
    clean = torch.where(known[:, None, :, None], clean, prior)
    return appearance, prior, known, trajectory(clean, prior, process)


@torch.no_grad()
def diagnostic(model, caches, device):
    from sklearn.metrics import average_precision_score
    values = [training_batch(c, 0, device, 29) for c in caches]
    app, prior, labels, known, nouns = [torch.cat(x) for x in zip(*values)]
    state = initialize(prior, 29)
    result = {}
    for name, appearance in [('factual', app), ('shuffled', app.roll(1, 0))]:
        prediction = torch.cat([model.reverse(state[s:s+32], appearance[s:s+32])
                               for s in range(0, len(state), 32)])
        probability = prediction.sum(1)[..., 0]
        truth = labels[known].cpu().numpy()
        score = probability[known].cpu().numpy()
        positive = known & (labels == 1)
        negative = known & (labels == 0)
        result[name] = {'micro_pair_ap': float(average_precision_score(truth, score)),
            'positive_probability': float(probability[positive].mean()),
            'negative_probability': float(probability[negative].mean()),
            'positive_recall_at_half': float((probability[positive] > .5).float().mean()),
            'negative_fpr_at_half': float((probability[negative] > .5).float().mean())}
    return result


def qualify(args, device):
    if args.output.exists():
        raise FileExistsError('Never overwrite qualification artifacts')
    args.output.mkdir(parents=True)
    ids = sorted(int(x) for x in (DATA / 'data/splits/vcoco_train.ids').read_text().split())[:32]
    caches = [load_cache(args.source_run, i) for i in ids]
    train, diagnostic_only = caches[:16], caches[16:]
    model = SliceDiT(81, 29, 777).to(device)
    checkpoint = args.source_run / 'checkpoint_final.pth'
    model.load_state_dict(torch.load(checkpoint, weights_only=False, map_location='cpu')['model'], strict=True)
    report = {'test_labels_read': False, 'train_image_ids': ids[:16], 'diagnostic_image_ids': ids[16:],
        'diagnostic_caveat': 'Disjoint repair samples, NOT unseen data: old checkpoint saw all trainval.',
        'initial_checkpoint_sha256': digest(checkpoint), 'source_sha256': source_hashes(),
        'before_fit': diagnostic(model.eval(), train, device),
        'before_disjoint': diagnostic(model, diagnostic_only, device)}
    process = ForwardDiffusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    started = time.monotonic()
    for update in range(64):
        model.train()
        app, prior, known, states = prepare_batch(train[update % len(train)], update // len(train), device, process)
        optimizer.zero_grad(set_to_none=True)
        summed, _ = backward_rollout(model, states, prior, app, known, known.sum())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(summed) or not torch.isfinite(norm):
            raise FloatingPointError('Closed-loop loss/gradient not finite')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Missing/nonfinite parameter gradient')
        optimizer.step()
        if update % 8 == 0:
            print(json.dumps({'qualification_update': update+1, 'loss': float(summed/known.sum().clamp_min(1)),
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
    report.update(after_fit=diagnostic(model.eval(), train, device),
                  after_disjoint=diagnostic(model, diagnostic_only, device),
                  seconds=time.monotonic()-started, peak_mib=torch.cuda.max_memory_allocated()/1024**2,
                  updates=64, samples_per_pair=10, steps=50)
    # A functional gate, not a paper accuracy claim or test-set search.
    report['status'] = 'PASS' if (report['after_fit']['factual']['micro_pair_ap'] >
                                report['before_fit']['factual']['micro_pair_ap'] and
                                report['after_fit']['factual']['micro_pair_ap'] >
                                report['after_fit']['shuffled']['micro_pair_ap']) else 'FAIL'
    write_json(args.output / 'qualification.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if report['status'] != 'PASS':
        raise RuntimeError('Repair did not pass the train-only closed-loop learning gate')


def train(args, device, rank, world):
    if world != 8:
        raise ValueError('Fixed effective global image batch8, eight ranks required')
    rows, channels = annotations()
    gate = json.loads(args.qualification.read_text())
    if gate['status'] != 'PASS' or gate['source_sha256'] != source_hashes():
        raise ValueError('Missing or stale train-only repair qualification')
    cache_summary = json.loads((args.source_run / 'cache_complete.json').read_text())
    if cache_summary['images'] != 5400:
        raise ValueError('Incomplete original train cache')
    protocol = {'architecture': 'unchanged SliceDiT-S + frozen official generic DDETR R50',
        'evidence_type': 'INDEPENDENT_CLOSED_LOOP_TRANSITION_REPAIR', 'author_equivalent': False,
        'source_sha256': source_hashes(), 'seed': 42, 'world_size': world, 'epochs': 30,
        'lr': 1e-4, 'weight_decay': 1e-4, 'precision': 'FP32', 'max_pairs_per_image': 8,
        'global_image_batch': 8, 'M': 10, 'K': 50, 'T': 2000,
        'initialization': 'fresh seed42; not qualification weights or previous failed checkpoint',
        'training': 'All50 reverse steps, own detached outputs, M sampled Eq4 trajectories; one/M deterministic-prior boundary',
        'objective': 'mean sampled previous-state MSE, known-slice global normalization; truncated BPTT1',
        'qualification_sha256': digest(args.qualification),
        'source_cache_run': str(args.source_run), 'source_cache_protocol_sha256': digest(args.source_run / 'protocol.json'),
        'source_cache_summary_sha256': digest(args.source_run / 'cache_complete.json'),
        'checkpoint_selection': 'fixed epoch30; no test-based selection',
        'test_labels_read_during_training': False,
        'limitations': ['not Eq6 posterior-density MSE', 'author DDETR identity unavailable',
                       'generic COCO detector overlap with VCOCO test not audited',
                       '29role/81noun native VCOCO adaptation is ours',
                       'one-step truncated BPTT and inference-boundary inclusion are independent choices']}
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != protocol or not (args.resume or args.evaluate_only):
                raise ValueError('Existing run requires exact unchanged explicit resume')
        else:
            if args.resume or args.evaluate_only:
                raise FileNotFoundError('Missing sealed run')
            write_json(path, protocol)
            shutil.copytree(Path(__file__).parent, args.output / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    dist.barrier()
    fingerprint = digest(args.output / 'protocol.json')
    model = SliceDiT(81, 29, 777).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cursor = {'epoch': 0, 'offset': 0, 'updates': 0}
    if args.resume:
        path = args.output / 'checkpoint_latest.pth'
        meta = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != meta['sha256']:
            raise ValueError('Resume checkpoint integrity mismatch')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != fingerprint or len(state['rng']) != world:
            raise ValueError('Resume protocol/world mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'][rank])
        cursor = state['cursor']
        del state
    wrapped = DDP(model, device_ids=[device.index])
    process = ForwardDiffusion().to(device)
    if not args.evaluate_only:
        for epoch in range(cursor['epoch'], 30):
            model.train()
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42+epoch)).tolist()[rank::world]
            start = cursor['offset'] if epoch == cursor['epoch'] else 0
            for offset in range(start, len(order)):
                begin = time.monotonic()
                cache = load_cache(args.source_run, rows[order[offset]]['image_id'])
                app, prior, known, states = prepare_batch(cache, epoch, device, process)
                count = known.sum().detach()
                dist.all_reduce(count)
                optimizer.zero_grad(set_to_none=True)
                summed, _ = backward_rollout(wrapped, states, prior, app, known, count, world_size=world)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm) or not torch.isfinite(summed):
                    raise FloatingPointError('Nonfinite closed-loop training')
                optimizer.step()
                dist.all_reduce(summed)
                cursor = {'epoch': epoch, 'offset': offset+1, 'updates': cursor['updates']+1}
                epoch_done = offset+1 == len(order)
                if epoch_done:
                    cursor.update(epoch=epoch+1, offset=0)
                peak = torch.tensor(torch.cuda.max_memory_allocated()/1024**2, device=device)
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if rank == 0:
                    record = {**cursor, 'display_epoch': epoch+1, 'target_epochs': 30,
                              'loss': float(summed/count.clamp_min(1)), 'gradient_norm': float(norm),
                              'peak_mib_max_rank': float(peak), 'seconds': time.monotonic()-begin,
                              'test_labels_read': False}
                    write_json(args.output / 'progress.json', record)
                    with (args.output / 'train.jsonl').open('a') as handle:
                        handle.write(json.dumps(record)+'\n')
                    if cursor['updates'] <= 3 or cursor['updates'] % 25 == 0:
                        print(json.dumps(record), flush=True)
                if cursor['updates'] == 1 or cursor['updates'] % 100 == 0 or epoch_done:
                    save_checkpoint(model, optimizer, args.output, cursor, fingerprint, rank, world)
        if rank == 0:
            shutil.copyfile(args.output / 'checkpoint_latest.pth', args.output / 'checkpoint_final.pth')
            write_json(args.output / 'training_complete.json', {'epochs': 30,
                'checkpoint_sha256': digest(args.output / 'checkpoint_final.pth'), 'protocol_sha256': fingerprint})
        dist.barrier()
    state = torch.load(args.output / 'checkpoint_final.pth', map_location='cpu', weights_only=False)
    if state['protocol_sha256'] != fingerprint or state['cursor']['epoch'] != 30:
        raise ValueError('Only fixed final epoch30 may be evaluated')
    if digest(args.output / 'checkpoint_final.pth') != json.loads((args.output / 'training_complete.json').read_text())['checkpoint_sha256']:
        raise ValueError('Final checkpoint hash mismatch')
    model.load_state_dict(state['model'], strict=True)
    del state, wrapped, optimizer
    torch.cuda.empty_cache()
    detector, loading = build_detector(device)
    write_json(args.output / f'detector_loading_rank{rank}.json', loading)
    evaluate(model, detector, channels, args.output, device, rank, world)
    if rank == 0:
        path = args.output / 'official_test/provenance.json'
        provenance = json.loads(path.read_text())
        provenance['evidence_type'] = protocol['evidence_type']
        write_json(path, provenance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--qualify', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    args.source_run, args.output = args.source_run.resolve(), args.output.resolve()
    if args.source_run == args.output:
        raise ValueError('Original failed run is immutable')
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.manual_seed(42+rank)
    random.seed(42+rank)
    np.random.seed(42+rank)
    if args.qualify:
        if world != 1:
            raise ValueError('Learning diagnostic is single-rank')
        qualify(args, device)
    else:
        if args.qualification is None:
            parser.error('--qualification is required for formal training')
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
        train(args, device, rank, world)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
