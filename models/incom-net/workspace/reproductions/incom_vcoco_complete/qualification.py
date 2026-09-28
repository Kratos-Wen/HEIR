"""Real-pretrained qualifications, separate from accuracy or author fidelity."""

import json
import socket

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from reproductions.incom_net.qualify import parameter_digest, check_ddp_normalization
from reproductions.incom_net.run import collate, rng_state, restore_rng
from .model import candidate_pairs, associate
from .run import atomic_json, move_batch


def selected_indices(data):
    def counts(i):
        people = data.annotations[i]['people']
        visible = sum(sum(v and y == 1 for v, y in zip(p['visible'], p['labels'])) for p in people)
        null = sum(sum(not v and y == 1 for v, y in zip(p['visible'][:25], p['labels'][:25])) for p in people)
        annotated = sum(any(y >= 0 for y in p['labels']) for p in people)
        return visible, null, annotated
    eligible = [i for i in range(len(data)) if all(c > 0 for c in counts(i)[:2])]
    return sorted(eligible, key=lambda i: (counts(i)[2], -min(counts(i)[:2]), data.annotations[i]['image_id']))


def qualify_cpu(model, data, output, fingerprint):
    path = output / 'cpu_qualification.json'
    if path.exists():
        raise FileExistsError('CPU qualification already exists')
    indices = selected_indices(data)[:2]
    if len(indices) != 2:
        raise ValueError('Insufficient original visible+null training examples')
    images, targets = collate([data[i] for i in indices])
    frozen = parameter_digest(model.extractor)
    with torch.no_grad():
        records = model.extractor(images)
    counts = {'visible': 0, 'null': 0, 'agent': 0}
    for r, t in zip(records, targets):
        pairs = candidate_pairs(r['labels'])
        y, _ = associate(r, pairs, t, model.compatibility)
        is_null = pairs[:, 1] == len(r['labels'])
        counts['visible'] += int(y[~is_null, :25].sum())
        counts['null'] += int(y[is_null, :25].sum())
        counts['agent'] += int(y[:, 25:].sum())
    if counts['visible'] == 0 or counts['null'] == 0:
        raise ValueError('Real pretrained detections did not exercise both visible and null supervision')
    model.train()
    model.zero_grad(set_to_none=True)
    result = model(images, targets)
    result['loss'].backward()
    active = [p for p in model.head.parameters() if p.requires_grad]
    if not torch.isfinite(result['loss']) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in active):
        raise FloatingPointError('Missing/nonfinite real-data gradients')
    if any(p.requires_grad or p.grad is not None for p in model.extractor.parameters()):
        raise ValueError('Frozen pretrained boundary violated')
    if frozen != parameter_digest(model.extractor):
        raise ValueError('Frozen weights changed')
    atomic_json(path, {'passed': True, 'protocol_sha256': fingerprint, 'strict_pretrained_load': True,
                'image_ids': [t['image_id'] for t in targets], 'matched': counts,
                'loss': float(result['loss'].detach()), 'all_head_gradients_finite': True,
                'frozen_sha256': frozen, 'test_labels_read': False,
                'author_fidelity_verified': False, 'accuracy_result': None})
    print(path.read_text(), flush=True)


def qualify_gpu(model, data, output, fingerprint, rank, world, device):
    if world != 8:
        raise ValueError('Qualification requires the actual eight-GPU configuration')
    gate = json.loads((output / 'cpu_qualification.json').read_text())
    if gate['protocol_sha256'] != fingerprint or not gate['passed']:
        raise ValueError('CPU qualification missing/stale')
    path = output / 'gpu_qualification.json'
    if path.exists():
        raise FileExistsError('Preserve existing GPU qualification')
    error = check_ddp_normalization(device, rank, world)
    order = sorted(selected_indices(data), key=lambda i: (
        -sum(any(y >= 0 for y in p['labels']) for p in data.annotations[i]['people']),
        -sum(sum(y == 1 for y in p['labels']) for p in data.annotations[i]['people']),
        data.annotations[i]['image_id']))
    if len(order) < 2 * world:
        raise ValueError('Too few original-label qualification samples')
    images, targets = move_batch(collate([data[i] for i in order[2*rank:2*rank+2]]), device)
    frozen = parameter_digest(model.extractor)
    wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=.2)
    torch.cuda.reset_peak_memory_stats(device)

    def step():
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        result = wrapped(images, targets)
        result['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), .1)
        if not torch.isfinite(result['loss']) or not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite real-data GPU loss/gradients')
        optimizer.step()
        return float(result['loss'].detach())

    first = step()
    scheduler.step()
    states = [None] * world
    dist.all_gather_object(states, rng_state())
    checkpoint = output / 'qualification_resume.pth'
    if rank == 0:
        torch.save({'head': model.head.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': states}, checkpoint)
    dist.barrier()
    uninterrupted = step()
    expected = {k: p.detach().cpu().clone() for k, p in model.head.state_dict().items()}
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.head.load_state_dict(state['head'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'][rank])
    replay = step()
    max_error = 0.
    for k, p in model.head.state_dict().items():
        torch.testing.assert_close(p.detach().cpu(), expected[k], atol=1e-6, rtol=1e-5)
        max_error = max(max_error, float((p.detach().cpu() - expected[k]).abs().max()))
    if abs(replay - uninterrupted) > 1e-4 * max(1., abs(uninterrupted)):
        raise ValueError('Resume loss mismatch')
    if parameter_digest(model.extractor) != frozen or any(p.grad is not None for p in model.extractor.parameters()):
        raise ValueError('Frozen weights changed')
    signatures = [None] * world
    dist.all_gather_object(signatures, parameter_digest(model.head))
    if len(set(signatures)) != 1:
        raise ValueError('Model parameters differ across ranks')
    memory = torch.cuda.max_memory_allocated(device)
    if memory >= .9 * torch.cuda.get_device_properties(device).total_memory:
        raise RuntimeError('Insufficient measured GPU memory headroom')
    record = {'rank': rank, 'host': socket.gethostname(), 'image_ids': [t['image_id'] for t in targets],
              'first_loss': first, 'uninterrupted_loss': uninterrupted, 'resume_loss': replay,
              'resume_max_error': max_error, 'gradient_normalization_max_error': error,
              'peak_allocated_bytes': memory, 'test_labels_read': False}
    records = [None] * world
    dist.all_gather_object(records, record)
    if rank == 0:
        atomic_json(path, {'passed': True, 'world_size': world, 'protocol_sha256': fingerprint,
                          'ranks': records, 'author_fidelity_verified': False, 'accuracy_result': None})
    dist.barrier()
