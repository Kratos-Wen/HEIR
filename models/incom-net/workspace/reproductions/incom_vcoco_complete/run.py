"""Preparation, fixed-epoch training and official test."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import filecmp
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from reproductions.incom_net.backbones import WORKSPACE, load_pretrained
from reproductions.incom_net.model import InCoMConfig
from reproductions.incom_net.runtime import RGBTest, collate, rng_state, restore_rng, DATA
from .data import CHANNELS, CompleteVCOCO, build_annotations
from .model import CompleteDetector, export_vcoco

SOURCE = Path(__file__).parent


def atomic_json(path, data):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def dataset(output, augment=False):
    return CompleteVCOCO(output / 'train_annotations.json', DATA / 'images/train2014', augment)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'protocol.json').exists():
        raise FileExistsError('This run is prepared; use a new run directory')
    inputs = [DATA / 'data/vcoco/vcoco_trainval.json', DATA / 'data/instances_vcoco_all_2014.json',
              DATA / 'data/splits/vcoco_trainval.ids']
    ids = [int(x) for x in inputs[2].read_text().split()]
    test_ids = {int(x) for x in (DATA / 'data/splits/vcoco_test.ids').read_text().split()}
    if len(ids) != len(set(ids)) or len(ids) != 5400 or len(test_ids) != 4946 or test_ids.intersection(ids):
        raise ValueError('Invalid official split IDs')
    data = build_annotations(json.loads(inputs[0].read_text()), json.loads(inputs[1].read_text()), ids)
    if len(data['compatibility']) != 80 or not data['counts'].get('null') or not data['counts'].get('visible'):
        raise ValueError('Incomplete original role ingestion')
    atomic_json(args.output / 'train_annotations.json', data)
    protocol = {'recipe': json.loads((SOURCE / 'recipe.json').read_text()),
                'config': asdict(InCoMConfig(num_actions=len(CHANNELS))), 'channels': list(CHANNELS),
                'detector': str(args.detector), 'clip': str(args.clip),
                'train_images': len(ids), 'label_counts': data['counts']}
    atomic_json(args.output / 'protocol.json', protocol)
    print(json.dumps({'prepared': str(args.output), 'counts': data['counts'], 'training_started': False}), flush=True)


def move_batch(batch, device):
    images, targets = batch
    return ([x.to(device) for x in images],
            [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets])


def ensure_milestone(latest, milestone):
    if milestone.exists():
        if not filecmp.cmp(latest, milestone, shallow=False):
            raise ValueError('Refuse to replace a different milestone checkpoint')
    else:
        os.link(latest, milestone)


def model_for(protocol, output, device):
    return CompleteDetector(load_pretrained(protocol['detector'], protocol['clip'], device),
                            InCoMConfig(**protocol['config']), dataset(output).compatibility).to(device)


def train(model, output, resume, rank, world, device):
    if world != 8:
        raise ValueError('Training is fixed at eight GPUs x batch two')
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=.2)
    data = dataset(output, True)
    sampler = DistributedSampler(data, world, rank, seed=42, drop_last=True)
    loader = DataLoader(data, batch_size=2, sampler=sampler, drop_last=True,
                        collate_fn=collate, num_workers=2, pin_memory=True)
    history = []
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if len(state['rng']) != world:
            raise ValueError('Resume checkpoint does not match the world size')
        model.head.load_state_dict(state['head'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'][rank])
        history = state['history']
        if [r['epoch'] for r in history] != list(range(1, state['epoch'] + 1)):
            raise ValueError('Invalid checkpoint history')
        if (output / 'checkpoint_latest.pth').exists() and not filecmp.cmp(resume, output / 'checkpoint_latest.pth', shallow=False):
            raise ValueError('Refuse rollback over a later checkpoint; use the latest checkpoint')
    elif (output / 'checkpoint_latest.pth').exists():
        raise FileExistsError('Explicit --resume required')
    if rank == 0 and resume and len(history) in (10, 20, 30):
        # Recover a crash between the atomic checkpoint save and milestone link.
        ensure_milestone(resume, output / f'checkpoint_epoch_{len(history):03d}.pth')
    dist.barrier()
    wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    for epoch in range(len(history), 30):
        sampler.set_epoch(epoch)
        wrapped.train()
        totals = torch.zeros(3, device=device)
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            result = wrapped(*move_batch(batch, device))
            if not torch.isfinite(result['loss']):
                raise FloatingPointError('Nonfinite training loss')
            result['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), .1)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            totals += torch.stack((result['loss'].detach(), result['matched_positive_edges'], totals.new_tensor(1)))
            if rank == 0 and step % 25 == 0:
                print(f'epoch={epoch+1}/30 step={step+1}/{len(loader)} loss={float(result["loss"]):.6f}', flush=True)
        scheduler.step()
        dist.all_reduce(totals)
        history.append({'epoch': epoch + 1, 'loss': float(totals[0] / totals[2]),
                        'matched_edges': float(totals[1]), 'lr_next_epoch': scheduler.get_last_lr()[0],
                        'global_batch': 16, 'optimizer_steps': len(loader), 'test_labels_read': False})
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            latest = output / 'checkpoint_latest.pth'
            tmp = latest.with_suffix('.partial')
            torch.save({'head': model.head.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'rng': states, 'epoch': epoch + 1,
                        'history': history}, tmp)
            tmp.replace(latest)
            atomic_json(output / 'train_metrics.json', history)
            if epoch + 1 in (10, 20, 30):
                ensure_milestone(latest, output / f'checkpoint_epoch_{epoch+1:03d}.pth')
            print(json.dumps(history[-1]), flush=True)
        dist.barrier()


def evaluate(model, output, rank, world, device):
    checkpoint = output / 'checkpoint_epoch_030.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['epoch'] != 30:
        raise ValueError('Expected the final epoch 30 checkpoint')
    model.head.load_state_dict(state['head'], strict=True)
    model.eval()
    data = RGBTest()
    dest = output / 'official_test'
    dest.mkdir(exist_ok=True)
    if (dest / 'official_metrics.json').exists():
        raise FileExistsError('Preserve existing official metrics')
    loader = DataLoader(Subset(data, range(rank, len(data), world)), batch_size=1, collate_fn=collate, num_workers=2)
    ids, predictions = [], []
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            result = model([x.to(device) for x in images])[0]
            target = targets[0]
            predictions.extend(export_vcoco(result, target['image_id'], target['original_wh']))
            ids.append(target['image_id'])
            if step % 100 == 0:
                print(f'test rank={rank}: {step+1}/{len(loader)}', flush=True)
    shard = dest / f'rank{rank}.pkl'
    tmp = shard.with_suffix('.partial')
    with tmp.open('wb') as handle:
        pickle.dump({'ids': ids, 'records': predictions}, handle, protocol=4)
    tmp.replace(shard)
    if world > 1:
        dist.barrier()
    if rank == 0:
        coverage, predictions = [], []
        for r in range(world):
            with (dest / f'rank{r}.pkl').open('rb') as handle:
                part = pickle.load(handle)
            coverage.extend(part['ids'])
            predictions.extend(part['records'])
        if len(coverage) != len(set(coverage)) or set(coverage) != set(data.ids):
            raise ValueError('Incomplete/duplicate official test coverage')
        cache = dest / 'predictions.pkl'
        tmp = cache.with_suffix('.partial')
        with tmp.open('wb') as handle:
            pickle.dump(predictions, handle, protocol=4)
        tmp.replace(cache)
        atomic_json(dest / 'coverage.json', {'images': len(coverage), 'image_ids': sorted(coverage),
                    'zero_prediction_image_ids': sorted(set(coverage) - {p['image_id'] for p in predictions})})
        subprocess.run([sys.executable, '-m', 'reproductions.incom_vcoco_complete.evaluate',
            '--cache', str(cache), '--output-json', str(dest / 'official_metrics.json')], check=True)
    if world > 1:
        dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('prepare', 'train', 'test'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detector', type=Path, default=WORKSPACE.parent / 'pvic/checkpoints/detr-r50-vcoco.pth')
    p.add_argument('--clip', type=Path, default=WORKSPACE / 'assets/incom_net_reproduction/ViT-L-14-336px.pt')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.mode == 'prepare':
        prepare(args)
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    rank, world, local = (int(os.environ.get(k, v)) for k, v in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)])
    device = torch.device(f'cuda:{local}')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    try:
        model = model_for(protocol, args.output, device)
        if args.mode == 'train':
            train(model, args.output, args.resume, rank, world, device)
        evaluate(model, args.output, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
