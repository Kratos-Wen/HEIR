"""User-approved independent transition-regression reconstruction, seed42."""

import argparse
from datetime import timedelta
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import functional as TF

from .detector import ROOT, AUTHOR, WEIGHTS, digest, build_detector, extract
from .vcoco import DATA, ANNOTATIONS, annotations, pair_targets, select_pairs, regression_loss, positive_coverage
from .diffusion import ForwardDiffusion, ground_truth, initialize
from .transition import exact_transition
from .model import SliceDiT
from .native_eval import predict_records


def write_json(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def image_tensor(path, device):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    resized = TF.resize(image, 800, max_size=1333)
    value = TF.normalize(TF.to_tensor(resized), [.485, .456, .406], [.229, .224, .225])
    return value.to(device), width, height


def train_image_path(filename):
    parts = Path(filename).name.split('_')
    if len(parts) != 3 or parts[0] != 'COCO' or parts[1] not in ('train2014', 'val2014'):
        raise ValueError('Unrecognized official COCO image filename')
    return DATA / 'images' / parts[1] / filename


def prepare_cache(detector, rows, channels, output, fingerprint, device, rank, world):
    cache_dir = output / 'train_features'
    cache_dir.mkdir(exist_ok=True)
    records = []
    for offset in range(rank, len(rows), world):
        row = rows[offset]
        path = cache_dir / f"{row['image_id']:012d}.pth"
        meta = path.with_suffix('.json')
        if path.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if (record['protocol_sha256'] != fingerprint or record['image_id'] != row['image_id']
                    or digest(path) != record['sha256']):
                raise ValueError('Cache integrity/protocol mismatch')
        else:
            image_path = train_image_path(row['file_name'])
            tensor, width, height = image_tensor(image_path, device)
            values = extract(detector, tensor)
            labels, nouns = pair_targets(values, row, width, height, channels)
            values.update(labels=labels, target_nouns=nouns, image_id=row['image_id'], width=width, height=height)
            temp = path.with_suffix('.partial')
            torch.save(values, temp)
            temp.replace(path)
            total_positive = sum(p['labels'].count(1) for p in row['people'])
            record = {'image_id': row['image_id'], 'sha256': digest(path),
                'image_sha256': digest(image_path), 'protocol_sha256': fingerprint,
                'candidate_pairs': len(labels), 'positive_pairs': int((labels == 1).any(-1).sum()),
                'supervised_pairs': int((labels >= 0).any(-1).sum()),
                'annotated_positive_channels': total_positive,
                'positive_candidate_channel_assignments': int((labels == 1).sum()),
                'positive_edge_coverage': positive_coverage(values, row, width, height, labels),
                'note': 'Candidate assignments can repeat one GT edge; not a GT recall estimate.'}
            write_json(meta, record)
        records.append(record)
        if len(records) % 25 == 0 or len(records) == 1:
            print(json.dumps({'stage': 'detector_cache', 'rank': rank, 'images': len(records),
                              'rank_total': len(rows[rank::world])}), flush=True)
            write_json(output / f'cache_rank{rank}_progress.json', {'images': len(records), 'rank_total': len(rows[rank::world])})
    write_json(output / f'cache_rank{rank}.json', records)
    dist.barrier()
    if rank == 0:
        all_records = sum([json.loads((output / f'cache_rank{r}.json').read_text()) for r in range(world)], [])
        if len(all_records) != 5400 or {r['image_id'] for r in all_records} != {r['image_id'] for r in rows}:
            raise ValueError('Incomplete or duplicate train cache')
        write_json(output / 'cache_complete.json', {'images': 5400, 'protocol_sha256': fingerprint,
            'candidate_pairs': sum(x['candidate_pairs'] for x in all_records),
            'positive_pairs': sum(x['positive_pairs'] for x in all_records),
            'original_positive_edges': sum(x['positive_edge_coverage']['original_positive_edges'] for x in all_records),
            'covered_positive_edges': sum(x['positive_edge_coverage']['covered_positive_edges'] for x in all_records),
            'images_without_supervised_pair': [x['image_id'] for x in all_records if not x['supervised_pairs']],
            'test_labels_read': False})
    dist.barrier()


def training_batch(cache, epoch, device, channel_count):
    indices = select_pairs(cache['labels'], epoch, cache['image_id'])
    if not len(indices):
        return (torch.zeros(1, 777, device=device), torch.nn.functional.one_hot(
            torch.tensor([80], device=device), 81).float(), torch.zeros(1, channel_count, device=device),
            torch.full((1, channel_count), False, device=device), torch.tensor([80], device=device))
    return (cache['appearance'][indices].to(device), cache['prior'][indices].to(device),
        cache['labels'][indices].clamp_min(0).float().to(device), cache['labels'][indices].ge(0).to(device),
        cache['target_nouns'][indices].to(device))


def save_checkpoint(model, optimizer, output, cursor, fingerprint, rank, world):
    states = [None] * world
    dist.all_gather_object(states, rng_state())
    if rank == 0:
        path = output / 'checkpoint_latest.pth'
        temp = path.with_suffix('.partial')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'cursor': cursor, 'rng': states, 'protocol_sha256': fingerprint}, temp)
        temp.replace(path)
        write_json(output / 'checkpoint_latest.json', {'cursor': cursor, 'sha256': digest(path),
                                                       'protocol_sha256': fingerprint})
    dist.barrier()


def fit(model, rows, channels, args, fingerprint, device, rank, world):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cursor = {'epoch': 0, 'offset': 0, 'updates': 0}
    if args.resume:
        path = args.output / 'checkpoint_latest.pth'
        meta = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != meta['sha256']:
            raise ValueError('Resume checkpoint integrity failed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['protocol_sha256'] != fingerprint or len(state['rng']) != world:
            raise ValueError('Resume protocol/world mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'][rank])
        cursor = state['cursor']
    wrapped = DDP(model, device_ids=[device.index])
    process = ForwardDiffusion(steps=50, trials=2000).to(device)
    model.train()
    for epoch in range(cursor['epoch'], 30):
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(42 + epoch)).tolist()[rank::world]
        start = cursor['offset'] if epoch == cursor['epoch'] else 0
        for offset in range(start, len(order)):
            begin = time.monotonic()
            row = rows[order[offset]]
            cache = torch.load(args.output / 'train_features' / f"{row['image_id']:012d}.pth", weights_only=True)
            appearance, nouns_prior, labels, known, nouns = training_batch(cache, epoch, device, len(channels))
            appearance, nouns_prior, labels, known, nouns = [x.repeat_interleave(10, 0)
                                                            for x in (appearance, nouns_prior, labels, known, nouns)]
            clean = ground_truth(nouns, labels, 81)
            prior = initialize(nouns_prior, len(channels))
            clean = torch.where(known[:, None, :, None], clean, prior)
            steps = torch.randint(1, 51, (len(clean),), device=device)
            current, previous = exact_transition(clean, prior, steps, process)
            prediction = wrapped(current, steps, appearance)
            summed, count = regression_loss(prediction, previous, known)
            global_count = count.detach().clone()
            dist.all_reduce(global_count)
            loss = summed * world / global_count.clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite diffusion training loss/gradient')
            optimizer.step()
            totals = torch.stack((summed.detach(), count.float()))
            dist.all_reduce(totals)
            cursor = {'epoch': epoch, 'offset': offset + 1, 'updates': cursor['updates'] + 1}
            epoch_done = cursor['offset'] == len(order)
            if epoch_done:
                cursor['epoch'], cursor['offset'] = epoch + 1, 0
            peak = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                record = {**cursor, 'display_epoch': epoch + 1, 'target_epochs': 30,
                    'loss': float(totals[0] / totals[1].clamp_min(1)), 'gradient_norm': float(norm),
                    'supervised_slices': int(totals[1]), 'seconds': time.monotonic() - begin,
                    'peak_mib_max_rank': float(peak), 'test_labels_read': False}
                with (args.output / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
                write_json(args.output / 'progress.json', record)
                if cursor['updates'] <= 3 or cursor['updates'] % 25 == 0:
                    print(json.dumps(record), flush=True)
            if cursor['updates'] == 1 or cursor['updates'] % 200 == 0 or epoch_done:
                save_checkpoint(model, optimizer, args.output, cursor, fingerprint, rank, world)
    if rank == 0:
        shutil.copyfile(args.output / 'checkpoint_latest.pth', args.output / 'checkpoint_final.pth')
        write_json(args.output / 'training_complete.json', {'epochs': 30, 'protocol_sha256': fingerprint,
            'checkpoint_sha256': digest(args.output / 'checkpoint_final.pth'), 'native_test_ap': None})
    dist.barrier()


@torch.no_grad()
def evaluate(model, detector, channels, output, device, rank, world):
    ids = [int(x) for x in (DATA / 'data/splits/vcoco_test.ids').read_text().split()]
    if len(ids) != 4946 or len(set(ids)) != 4946:
        raise ValueError('Invalid official test IDs')
    model.eval()
    directory = output / 'official_test'
    directory.mkdir(exist_ok=True)
    records, covered, empty = [], [], []
    for image_id in ids[rank::world]:
        tensor, width, height = image_tensor(DATA / 'images/val2014' / f'COCO_val2014_{image_id:012d}.jpg', device)
        cache = extract(detector, tensor)
        prediction = predict_records(model, cache, channels, image_id, width, height, device)
        records.extend(prediction)
        covered.append(image_id)
        if not prediction:
            empty.append(image_id)
        if len(covered) % 50 == 0:
            print(json.dumps({'stage': 'test_inference', 'rank': rank, 'images': len(covered)}), flush=True)
    path = directory / f'rank{rank}.pkl'
    with path.with_suffix('.partial').open('wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    path.with_suffix('.partial').replace(path)
    write_json(directory / f'coverage_rank{rank}.json', {'image_ids': covered, 'empty_image_ids': empty})
    dist.barrier()
    if rank != 0:
        return
    records, covered, empty = [], [], []
    for r in range(world):
        with (directory / f'rank{r}.pkl').open('rb') as f:
            records.extend(pickle.load(f))
        coverage = json.loads((directory / f'coverage_rank{r}.json').read_text())
        covered.extend(coverage['image_ids'])
        empty.extend(coverage['empty_image_ids'])
    if len(covered) != 4946 or set(covered) != set(ids):
        raise ValueError('Missing or duplicate test coverage')
    cache_path = directory / 'vcoco_predictions.pkl'
    with cache_path.open('wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(directory / 'coverage.json', {'image_ids': sorted(covered), 'empty_image_ids': sorted(empty), 'images': 4946})
    command = [sys.executable, str(ROOT / 'vcoco_eval/scripts/evaluate_vcoco_official.py'),
        '--cache', str(cache_path), '--vsrl-json', str(DATA / 'data/vcoco/vcoco_test.json'),
        '--coco-json', str(DATA / 'data/instances_vcoco_all_2014.json'),
        '--split-ids', str(DATA / 'data/splits/vcoco_test.ids'), '--evaluator', str(DATA / 'vsrl_eval.py'),
        '--output-json', str(directory / 'official_metrics.json'), '--agent-ap']
    with (directory / 'official_eval.log').open('w') as f:
        subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    write_json(directory / 'provenance.json', {'evidence_type': 'USER_APPROVED_INDEPENDENT_TRANSITION_REGRESSION',
        'paper_equivalence': False, 'detector_pretraining_overlap_audit_complete': False,
        'cache_sha256': digest(cache_path), 'checkpoint_sha256': digest(output / 'checkpoint_final.pth')})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args()
    args.output = args.output.resolve()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8:
        raise ValueError('Fixed global batch eight: eight ranks, one image each')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    rows, channels = annotations()
    sources = list(Path(__file__).parent.glob('*.py')) + list(AUTHOR.rglob('*.py'))
    sources += [ROOT / 'reference_repos/DiT/models.py', WEIGHTS, ANNOTATIONS,
                ROOT / 'vcoco_eval/scripts/train_idiff_existing_2node8gpu.sh']
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        gate_path = ROOT / 'vcoco_eval/reports/idiff_native_gpu_qualification_20260919_r3.json'
        gate = json.loads(gate_path.read_text())
        if gate['status'] != 'PASS' or any(digest(Path(name)) != value for name, value in gate['source_sha256'].items()):
            raise ValueError('Native GPU qualification does not cover the current implementation')
        protocol = {'evidence_type': 'USER_APPROVED_INDEPENDENT_TRANSITION_REGRESSION',
            'gpu_qualification_sha256': digest(gate_path),
            'paper_equivalence': False, 'seed': 42, 'world_size': 8, 'global_image_batch': 8,
            'epochs': 30, 'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 1e-4,
            'lr_schedule': 'constant; paper schedule unavailable', 'precision': 'FP32',
            'architecture': 'official DiT-S blocks, width384/depth12/heads6, slice patchification',
            'nouns': 81, 'channels': channels, 'max_pairs_per_image_update': 8,
            'samples_per_pair': 10, 'diffusion_steps': 50, 'multinomial_trials': 2000,
            'beta': [.001, .2], 'checkpoint_selection': 'fixed final epoch30, no test selection',
            'objective': 'MSE to sampled exact-trajectory previous state; NOT Eq6 posterior-density MSE',
            'train_images': 5400, 'test_labels_read_during_training': False,
            'independent_choices': ['generic frozen official DDETR-R50, not author-identical task asset',
                'author reference attention on CUDA; no new detector implementation',
                'normalized sigmoid COCO noun prior; explicit 91-to-80 sparse ID mapping',
                'pair appearance = human query + entity query + projected union ROI + boxes + null flag',
                'detector NMS0.6, score0.2, max10 humans/30 entities, one virtual-null pair per human',
                'IoU>=0.5 positive, <0.3 negative, unknown and ambiguous labels masked',
                'eight sampled pairs/image (up to four preferred positives), epoch-seeded rotation',
                '29 native action-role channels and null noun preserve point/agent-only/missing roles',
                'sampled previous-state MSE replaces underspecified density target with user approval',
                'fixed30 epochs, constant LR, no augmentation of cached frozen features',
                'rank score = selected noun/presence joint probability * detector pair confidence'],
            'pretraining_exposure': 'Generic COCO pretrained detector; exact V-COCO test-image detection-pretraining overlap not yet audited. Not a matched detector comparison.',
            'source_sha256': {str(f): digest(f) for f in sorted(set(sources))},
            'torch': torch.__version__, 'cuda': torch.version.cuda}
        path = args.output / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != protocol or not (args.resume or args.evaluate_only):
                raise ValueError('Existing run requires unchanged explicit resume/evaluate protocol')
        else:
            if args.resume or args.evaluate_only:
                raise FileNotFoundError('Missing sealed run')
            write_json(path, protocol)
            shutil.copytree(Path(__file__).parent, args.output / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    dist.barrier()
    fingerprint = digest(args.output / 'protocol.json')
    detector, loading = build_detector(device)
    write_json(args.output / f'detector_loading_rank{rank}.json', loading)
    if not args.evaluate_only:
        prepare_cache(detector, rows, channels, args.output, fingerprint, device, rank, world)
    # Release detector memory during diffusion training; reconstruct strictly for test.
    detector.cpu()
    torch.cuda.empty_cache()
    model = SliceDiT(81, len(channels), 777).to(device)
    if not args.evaluate_only:
        fit(model, rows, channels, args, fingerprint, device, rank, world)
    state = torch.load(args.output / 'checkpoint_final.pth', map_location='cpu', weights_only=False)
    if state['protocol_sha256'] != fingerprint or state['cursor']['epoch'] != 30:
        raise ValueError('Only fixed final checkpoint is eligible for test')
    model.load_state_dict(state['model'], strict=True)
    del state
    detector.to(device)
    evaluate(model, detector, channels, args.output, device, rank, world)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
