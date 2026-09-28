"""Sealed, fixed-budget native-24 reconstruction training and official test.

This entry point never requests resources or launches a watcher. Run training
under torchrun within an existing allocation. Preparation requires explicit
acknowledgment of the unresolved author conventions in reconstruction.json.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from datetime import timedelta

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torchvision.transforms import functional as TF

from .backbones import WORKSPACE, PVIC, CLIP, digest, load_pretrained
from .model import InCoMConfig
from .vcoco import NativeVCOCO, InCoMDetector, export_vcoco

DATA = WORKSPACE.parent / 'data/v-coco'
SOURCE = Path(__file__).parent


def atomic_json(path, payload):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def collate(batch):
    return tuple(map(list, zip(*batch)))


def training_data(augment):
    return NativeVCOCO(PVIC / 'vcoco/instances_vcoco_trainval.json', DATA / 'images/train2014',
                       DATA / 'data/splits/vcoco_trainval.ids', augment=augment)


def contract(args):
    files = list(SOURCE.glob('*.py')) + [SOURCE / 'reconstruction.json',
             PVIC / 'vcoco/instances_vcoco_trainval.json', DATA / 'data/splits/vcoco_trainval.ids',
             args.detector, args.clip, WORKSPACE / 'vcoco_eval/scripts/evaluate_vcoco_official.py',
             DATA / 'vsrl_eval.py']
    files += list((PVIC / 'detr').rglob('*.py')) + list(CLIP.rglob('*.py'))
    spec = json.loads((SOURCE / 'reconstruction.json').read_text())
    return {'evidence_type': spec['evidence_type'], 'implementation_spec': spec,
            'config': asdict(InCoMConfig()), 'seed': 42, 'world_size': 8,
            'nominal_global_batch': 16, 'precision': 'fp32', 'epochs': 30,
            'checkpoint_selection': 'fixed epoch 30, no intermediate test',
            'frozen_asset_paths': {'detector': str(args.detector), 'clip': str(args.clip)},
            'input_and_source_sha256': {str(p): digest(p) for p in sorted(set(files))},
            'torch': torch.__version__}


class RGBTest(Dataset):
    def __init__(self):
        self.ids = [int(x) for x in (DATA / 'data/splits/vcoco_test.ids').read_text().split()]
        if len(self.ids) != 4946 or len(set(self.ids)) != 4946:
            raise ValueError('Invalid official test ID set')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(DATA / 'images/val2014' / f'COCO_val2014_{image_id:012d}.jpg').convert('RGB')
        original_wh = image.size
        image = TF.normalize(TF.to_tensor(TF.resize(image, 800, max_size=1333)),
                              [.485, .456, .406], [.229, .224, .225])
        return image, {'image_id': image_id, 'original_wh': original_wh}


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(rng):
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(rng['cuda'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])


def train(model, args, protocol_sha, rank, world, device):
    if world != 8:
        raise ValueError('Sealed training requires eight GPUs, per-GPU batch two')
    dataset = training_data(True)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=42, drop_last=True)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=collate,
                        num_workers=2, pin_memory=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=.2)
    start = 0
    latest = args.output / 'checkpoint_latest.pth'
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != protocol_sha or len(state['rng']) != world:
            raise ValueError('Checkpoint protocol/world size mismatch')
        model.head.load_state_dict(state['head'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng = state['rng'][rank]
        restore_rng(rng)
        start = state['epoch']
    elif latest.exists():
        raise FileExistsError('Explicit --resume required; do not overwrite a training run')
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    for epoch in range(start, 30):
        sampler.set_epoch(epoch)
        wrapped.train()
        totals = torch.zeros(3, device=device)
        for step, (images, targets) in enumerate(loader):
            images = [x.to(device) for x in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = wrapped(images, targets)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite MFT loss')
            losses['loss'].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), .1)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite MFT gradient')
            optimizer.step()
            totals += torch.stack((losses['loss'].detach(), losses['matched_positive_edges'], totals.new_tensor(1)))
            if rank == 0 and step % 25 == 0:
                print(f'epoch={epoch + 1}/30 step={step + 1}/{len(loader)} loss={float(losses["loss"]):.6f}', flush=True)
        scheduler.step()
        dist.all_reduce(totals)
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        if rank == 0:
            record = {'epoch': epoch + 1, 'loss': float(totals[0] / totals[2]),
                      'matched_edges': float(totals[1]), 'test_read': False,
                      'lr': optimizer.param_groups[0]['lr']}
            with (args.output / 'train.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            temporary = latest.with_suffix('.partial')
            torch.save({'head': model.head.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch + 1,
                        'protocol_sha256': protocol_sha, 'rng': states}, temporary)
            temporary.replace(latest)
            if (epoch + 1) % 10 == 0:
                milestone = args.output / f'checkpoint_epoch_{epoch + 1:03d}.pth'
                os.link(latest, milestone)
            print(json.dumps(record), flush=True)
        dist.barrier()


def test(model, args, protocol_sha, rank, world, device):
    checkpoint_path = args.output / 'checkpoint_epoch_030.pth'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if state['protocol_sha256'] != protocol_sha or state['epoch'] != 30:
        raise ValueError('Official test only accepts the predeclared final checkpoint')
    model.head.load_state_dict(state['head'], strict=True)
    model.eval()
    dataset = RGBTest()
    loader = DataLoader(Subset(dataset, list(range(rank, len(dataset), world))), batch_size=1,
                        num_workers=2, collate_fn=collate)
    output = args.output / 'official_test'
    output.mkdir(exist_ok=True)
    records, ids = [], []
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            predictions = model([x.to(device) for x in images])
            for prediction, target in zip(predictions, targets):
                records.extend(export_vcoco(prediction, target['image_id'], target['original_wh']))
                ids.append(target['image_id'])
            if step % 100 == 0:
                print(f'test rank={rank} {step + 1}/{len(loader)}', flush=True)
    shard = output / f'rank{rank}.pkl'
    temporary = shard.with_suffix('.partial')
    with temporary.open('wb') as handle:
        pickle.dump({'ids': ids, 'records': records}, handle, protocol=4)
    temporary.replace(shard)
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged, coverage = [], []
        for i in range(world):
            with (output / f'rank{i}.pkl').open('rb') as handle:
                part = pickle.load(handle)
            merged.extend(part['records'])
            coverage.extend(part['ids'])
        if len(coverage) != len(set(coverage)) or set(coverage) != set(dataset.ids):
            raise ValueError('Incomplete or duplicated test inference coverage')
        cache = output / 'predictions.pkl'
        temporary = cache.with_suffix('.partial')
        with temporary.open('wb') as handle:
            pickle.dump(merged, handle, protocol=4)
        temporary.replace(cache)
        atomic_json(output / 'coverage.json', {'images': len(coverage), 'image_ids': sorted(coverage),
                    'checkpoint_sha256': digest(checkpoint_path), 'protocol_sha256': protocol_sha})
        subprocess.run([sys.executable, str(WORKSPACE / 'vcoco_eval/scripts/evaluate_vcoco_official.py'),
                        '--cache', str(cache), '--vsrl-json', str(DATA / 'data/vcoco/vcoco_test.json'),
                        '--coco-json', str(DATA / 'data/instances_vcoco_all_2014.json'),
                        '--split-ids', str(DATA / 'data/splits/vcoco_test.ids'),
                        '--evaluator', str(DATA / 'vsrl_eval.py'), '--output-json', str(output / 'official_metrics.json'),
                        '--iou-thr', '0.5'], check=True)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'qualify', 'train', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector', type=Path, default=WORKSPACE.parent / 'pvic/checkpoints/detr-r50-vcoco.pth')
    parser.add_argument('--clip', type=Path, default=WORKSPACE / 'assets/incom_net_reproduction/ViT-L-14-336px.pt')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--cpu-qualification', type=Path, default=WORKSPACE / 'vcoco_eval/reports/incom_net_pretrained_gt_conformance_20260918.json')
    parser.add_argument('--acknowledge-unofficial', action='store_true')
    args = parser.parse_args()
    if not args.acknowledge_unofficial:
        parser.error('Read reconstruction.json and explicitly acknowledge the unofficial native-24 assumptions')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    current = contract(args)
    if args.mode == 'prepare':
        if protocol_path.exists():
            raise FileExistsError('A sealed protocol already exists; do not reseal a running experiment')
        qualification_path = args.cpu_qualification
        qualification = json.loads(qualification_path.read_text())
        if (not qualification['target_kind'].startswith('actual V-COCO')
                or qualification['matched_positive_edges'] < 1
                or not qualification['all_trainable_gradients_finite']
                or not qualification['strict_pretrained_load']
                or qualification['reconstruction_spec_sha256'] != digest(SOURCE / 'reconstruction.json')):
            raise ValueError('Current implementation lacks a real-pretrained, real-label qualification')
        for name in ('model.py', 'backbones.py', 'vcoco.py', 'validate.py'):
            if qualification['source_sha256'][name] != digest(SOURCE / name):
                raise ValueError(f'{name} changed after real-pretrained qualification')
        if (qualification['detector_sha256'] != digest(args.detector)
                or qualification['clip_sha256'] != digest(args.clip)):
            raise ValueError('Qualified weights differ from the requested training initialization')
        data = training_data(False)
        for row in data.annotations:
            if not len(row['actions']) == len(row['objects']) == len(row['boxes_h']) == len(row['boxes_o']):
                raise ValueError('Misaligned training annotation arrays')
        current['training_images'] = len(data)
        atomic_json(protocol_path, current)
        print(json.dumps({'prepared': str(protocol_path), 'train_images': len(data), 'training_started': False}))
        return
    saved = json.loads(protocol_path.read_text())
    current['training_images'] = len(training_data(False))
    if saved != current:
        raise ValueError('Source, data, assets or recipe changed since preparation')
    rank, world, local = [int(os.environ.get(k, default)) for k, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')]]
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42 + rank)
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    device = torch.device('cuda', local)
    cfg = InCoMConfig()
    model = InCoMDetector(load_pretrained(args.detector, args.clip, device), cfg,
                          training_data(False).compatibility).to(device)
    fingerprint = digest(protocol_path)
    if args.mode == 'qualify':
        from .qualify import qualify
        qualify(model, args, fingerprint, rank, world, device)
        dist.destroy_process_group()
        return
    if args.mode == 'train':
        gate = json.loads((args.output / 'qualification' / 'passed.json').read_text())
        if gate['protocol_sha256'] != fingerprint or not gate['passed'] or gate['world_size'] != world:
            raise ValueError('Current sealed protocol has no matching GPU/DDP qualification')
        train(model, args, fingerprint, rank, world, device)
    test(model, args, fingerprint, rank, world, device)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
