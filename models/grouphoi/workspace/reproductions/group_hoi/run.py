"""Fixed-final-epoch GroupHOI-S reconstruction; no test labels during training."""

import argparse
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .runtime import (AUTHOR, ASSETS, ROOT, DATA, sha, build_runtime, load_detr,
                      optimizer, train_dataset, batch_to_device, total_loss)
from .export import cache_records
from .postprocess import GroupCachePostprocessor


def atomic_json(path, payload):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


class RGBTest(Dataset):
    def __init__(self, preprocess):
        from datasets.vcoco import make_vcoco_transforms
        self.ids = [int(x) for x in (DATA / 'data/splits/vcoco_test.ids').read_text().split()]
        if len(self.ids) != 4946 or len(set(self.ids)) != 4946:
            raise ValueError('Incorrect test ID set')
        self.preprocess = preprocess
        self.transforms = make_vcoco_transforms('val')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(DATA / 'images/val2014' / f'COCO_val2014_{image_id:012d}.jpg').convert('RGB')
        w, h = image.size
        # Upstream evaluates CLIP on the resized PIL image, before normalization.
        resized, _ = self.transforms[0](image, None)
        tensor, _ = self.transforms[1](resized, None)
        return tensor, {'img_id': image_id, 'orig_size': torch.tensor([h, w]),
                        'clip_inputs': self.preprocess(resized)}


def contract(args, config):
    sources = list(Path(__file__).parent.glob('*.py')) + list(AUTHOR.rglob('*.py'))
    sources += list((ROOT / 'reference_repos/CLIP/clip').rglob('*.py'))
    inputs = [DATA / 'annotations/trainval_vcoco.json', DATA / 'data/splits/vcoco_trainval.ids',
              ASSETS / 'ViT-B-16.pt', ASSETS / 'detr-r50-e632da11.pth']
    return {'evidence_type': 'OFFICIAL_GROUPHOI_CORE_WITH_DECLARED_INTERFACE_RECONSTRUCTION',
            'author_equivalent_verified': False, 'config': vars(config),
            'world_size': 2, 'effective_batch': 8, 'precision': 'fp32', 'seed': 42,
            'checkpoint_selection': 'fixed epoch 90; never select on test',
            'sha256': {str(p): sha(p) for p in sorted(set(sources + inputs))},
            'torch': torch.__version__, 'runtime_changes': [
                'Explicit frozen OpenAI dense CLIP interface, unavailable author LAVIS fork.',
                'QBC->BQC bridge for Hungarian matcher only; native losses unchanged.',
                'DETR load validates all existing DETR layers; original converter unused keys recorded.',
                'No training-time upstream val (which is actually official test).',
                'CuDNN deterministic; full deterministic-algorithm enforcement disabled for CUDA indexing backward.']}


def fit(model, criterion, vlm, config, vis, txt, args, device, rank, world, fingerprint):
    dataset = train_dataset(vis, txt, config)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=42)
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True,
                        num_workers=2, pin_memory=True, collate_fn=__import__('util.misc', fromlist=['collate_fn']).collate_fn)
    opt, names = optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=.1)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != fingerprint or len(state['rng']) != world:
            raise ValueError('Resume contract/world mismatch')
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'][rank])
        start = state['epoch']
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    if rank == 0:
        atomic_json(args.output / 'optimizer_groups.json', names)
    for epoch in range(start, 90):
        sampler.set_epoch(epoch)
        wrapped.train()
        totals = torch.zeros(2, device=device)
        begin = time.monotonic()
        for step, batch in enumerate(loader):
            samples, targets, clip = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            output = wrapped(samples, vlm, clip_input=clip)
            loss, terms = total_loss(criterion, output, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            opt.step()
            totals += torch.stack((loss.detach(), loss.new_tensor(1)))
            if rank == 0 and step % 25 == 0:
                print(f'epoch={epoch+1}/90 step={step+1}/{len(loader)} loss={float(loss):.6f}', flush=True)
        scheduler.step()
        dist.all_reduce(totals)
        rngs = [None] * world
        dist.all_gather_object(rngs, rng_state())
        if rank == 0:
            latest = args.output / 'checkpoint_latest.pth'
            temporary = latest.with_suffix('.partial')
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(), 'rng': rngs,
                        'epoch': epoch+1, 'protocol_sha256': fingerprint}, temporary)
            temporary.replace(latest)
            if (epoch+1) % 30 == 0:
                os.link(latest, args.output / f'checkpoint_epoch_{epoch+1:03d}.pth')
            record = {'epoch': epoch+1, 'loss': float(totals[0]/totals[1]),
                      'seconds': time.monotonic()-begin, 'test_read': False,
                      'lr': [g['lr'] for g in opt.param_groups]}
            with (args.output / 'train.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
        dist.barrier()


@torch.no_grad()
def infer(model, vlm, config, vis, args, device, rank, world):
    from util.misc import collate_fn
    dataset = RGBTest(vis['eval'])
    subset = torch.utils.data.Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(subset, batch_size=2, num_workers=2, collate_fn=collate_fn)
    processor = GroupCachePostprocessor(config, np.load(DATA / 'annotations/corre_vcoco.npy')).to(device)
    model.eval()
    records, ids, empty = [], [], []
    output = args.output / 'official_test'
    output.mkdir(exist_ok=True)
    for step, batch in enumerate(loader):
        samples, targets, clip = batch_to_device(batch, device)
        prediction = model(samples, vlm, is_training=False, clip_input=clip)
        results = processor(prediction, torch.stack([t['orig_size'] for t in targets]))
        for result, target in zip(results, targets):
            image_id = int(target['img_id'])
            rows = cache_records(result, image_id)
            records.extend(rows)
            ids.append(image_id)
            if not rows:
                empty.append(image_id)
        if step % 50 == 0:
            print(f'test rank={rank} images={len(ids)}/{len(subset)}', flush=True)
    path = output / f'predictions_rank{rank}.pkl'
    with path.with_suffix('.partial').open('wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    path.with_suffix('.partial').replace(path)
    atomic_json(output / f'coverage_rank{rank}.json', {'image_ids': ids, 'empty_image_ids': empty})
    dist.barrier()
    if rank != 0:
        return
    records, ids, empty = [], [], []
    for r in range(world):
        with (output / f'predictions_rank{r}.pkl').open('rb') as f:
            records.extend(pickle.load(f))
        coverage = json.loads((output / f'coverage_rank{r}.json').read_text())
        ids += coverage['image_ids']
        empty += coverage['empty_image_ids']
    if len(ids) != 4946 or set(ids) != set(dataset.ids):
        raise ValueError('Test coverage mismatch')
    cache = output / 'vcoco_predictions.pkl'
    with cache.open('wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_json(output / 'coverage.json', {'image_ids': sorted(ids), 'images': len(ids), 'empty_image_ids': sorted(empty)})
    command = [sys.executable, str(ROOT / 'vcoco_eval/scripts/evaluate_vcoco_official.py'),
               '--cache', str(cache), '--vsrl-json', str(DATA / 'data/vcoco/vcoco_test.json'),
               '--coco-json', str(DATA / 'data/instances_vcoco_all_2014.json'),
               '--split-ids', str(DATA / 'data/splits/vcoco_test.ids'), '--evaluator', str(DATA / 'vsrl_eval.py'),
               '--output-json', str(output / 'official_metrics.json'), '--agent-ap']
    with (output / 'official_eval.log').open('w') as f:
        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    atomic_json(output / 'provenance.json', {'evidence_type': 'OFFICIAL_CORE_WITH_INTERFACE_RECONSTRUCTION',
                'author_equivalent_verified': False, 'checkpoint_sha256': sha(args.checkpoint or args.output / 'checkpoint_epoch_090.pth'),
                'cache_sha256': sha(cache), 'metrics_sha256': sha(output / 'official_metrics.json')})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--checkpoint', type=Path, help='Strict inference-only checkpoint')
    args = p.parse_args()
    args.output = args.output.resolve()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 2:
        raise ValueError('Preserve author BatchNorm batch: two GPUs x four images')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.manual_seed(42+rank)
    np.random.seed(42+rank)
    random.seed(42+rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl')
    model, criterion, vlm, config, vis, txt = build_runtime(device)
    loading = load_detr(model)
    payload = contract(args, config)
    # Device index is rank-local, not a protocol difference.
    payload['config']['device'] = 'cuda'
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        sealed = args.output / 'protocol.json'
        if sealed.exists():
            if json.loads(sealed.read_text()) != payload:
                raise ValueError('Sealed protocol changed')
            if not args.resume and not args.checkpoint:
                raise FileExistsError('Explicit resume required')
        else:
            atomic_json(sealed, payload)
            shutil.copytree(Path(__file__).parent, args.output / 'source', ignore=shutil.ignore_patterns('__pycache__'))
            atomic_json(args.output / 'initialization.json', loading)
    dist.barrier()
    fingerprint = sha(args.output / 'protocol.json')
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != fingerprint or state['epoch'] != 90:
            raise ValueError('Only sealed fixed-final checkpoint is eligible')
        model.load_state_dict(state['model'], strict=True)
    else:
        fit(model, criterion, vlm, config, vis, txt, args, device, rank, world, fingerprint)
        # DDP broadcasts BN buffers before a forward, but local final running
        # statistics can differ afterwards. Evaluate the one saved checkpoint.
        state = torch.load(args.output / 'checkpoint_epoch_090.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
    infer(model, vlm, config, vis, args, device, rank, world)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
