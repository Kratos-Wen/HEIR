"""Eight-rank numerical and real-train-data qualification; never an AP run."""

import json
import importlib.util
import hashlib
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .model import focal_mft_loss
from .run import DATA, atomic_json, rng_state, restore_rng, training_data


def check_ddp_normalization(device, rank, world):
    errors = []
    for positive_count in (0, 1, 3, 16):
        torch.manual_seed(123)
        layer = torch.nn.Linear(4, 24).to(device)
        reference = torch.nn.Linear(4, 24).to(device)
        reference.load_state_dict(layer.state_dict())
        wrapped = DDP(layer, device_ids=[device.index])
        x = torch.arange(64, device=device).reshape(16, 4).float() / 64
        y = torch.zeros(16, 24, device=device)
        y[:positive_count, 0] = 1
        sl = slice(2 * rank, 2 * rank + 2)
        denominator = y[sl].sum()
        dist.all_reduce(denominator)
        denominator = denominator.clamp(min=1) / world
        logits = wrapped(x[sl])
        modes = ('full', 'detector_only', 'vlm_only')
        loss, _ = focal_mft_loss({k: {'logits': logits} for k in modes}, y[sl], y[sl].bool() | True,
                                alpha=.5, gamma=.1, normalizer=denominator)
        loss.backward()
        pred = reference(x)
        global_loss, _ = focal_mft_loss({k: {'logits': pred} for k in modes}, y, y.bool() | True,
                                       alpha=.5, gamma=.1)
        global_loss.backward()
        for a, b in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-5)
            errors.append(float((a.grad - b.grad).abs().max()))
        del wrapped, layer, reference
    return max(errors)


def parameter_digest(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def qualify(model, args, fingerprint, rank, world, device):
    if world != 8:
        raise ValueError('Qualification requires the actual eight-GPU topology')
    output = args.output / 'qualification'
    output.mkdir(exist_ok=True)
    if (output / 'passed.json').exists():
        raise FileExistsError('Qualification already exists; preserve it')
    started = time.monotonic()
    # Import the actual evaluator without reading any test labels.
    spec = importlib.util.spec_from_file_location('incom_vcoco_official', DATA / 'vsrl_eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    assert callable(evaluator.VCOCOeval)
    normalization_error = check_ddp_normalization(device, rank, world)
    torch.manual_seed(42 + rank)
    dataset = training_data(False)
    # Fixed annotation-only selection, including the highest person-count scenes.
    order = sorted(range(len(dataset)), key=lambda i: (
        -len(set(tuple(b) for b in dataset.annotations[i]['boxes_h'])),
        -len(dataset.annotations[i]['actions']), dataset.annotations[i]['file_name']))
    indices = order[rank * 2:rank * 2 + 2]
    batch = [dataset[i] for i in indices]
    images = [x.to(device) for x, _ in batch]
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for _, t in batch]
    frozen_hash = parameter_digest(model.extractor)
    wrapped = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=.2)
    torch.cuda.reset_peak_memory_stats(device)

    def step():
        wrapped.train()
        optimizer.zero_grad(set_to_none=True)
        losses = wrapped(images, targets)
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError('Real-label qualification loss is nonfinite')
        losses['loss'].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), .1)
        if not torch.isfinite(norm):
            raise FloatingPointError('Real-label qualification gradient is nonfinite')
        optimizer.step()
        return {k: float(v.detach()) for k, v in losses.items()}

    first = step()
    scheduler.step()
    states = [None] * world
    dist.all_gather_object(states, rng_state())
    checkpoint = output / 'resume_checkpoint.pth'
    if rank == 0:
        temporary = checkpoint.with_suffix('.partial')
        torch.save({'head': model.head.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': states,
                    'protocol_sha256': fingerprint}, temporary)
        temporary.replace(checkpoint)
    dist.barrier()
    uninterrupted = step()
    expected = {k: p.detach().cpu().clone() for k, p in model.head.state_dict().items()}
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.head.load_state_dict(saved['head'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'][rank])
    replay = step()
    max_error = 0.
    for name, p in model.head.state_dict().items():
        actual = p.detach().cpu()
        torch.testing.assert_close(actual, expected[name], atol=1e-6, rtol=1e-5)
        max_error = max(max_error, float((actual - expected[name]).abs().max()))
    assert abs(uninterrupted['loss'] - replay['loss']) <= 1e-4 * max(1., abs(uninterrupted['loss']))
    assert all(not p.requires_grad and p.grad is None for p in model.extractor.parameters())
    assert parameter_digest(model.extractor) == frozen_hash
    signature = torch.stack([p.detach().double().sum() for p in model.head.parameters()])
    low, high = signature.clone(), signature.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    torch.testing.assert_close(low, high, atol=1e-8, rtol=1e-8)
    allocated = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    if allocated > .90 * total:
        raise RuntimeError('Dense-scene batch leaves less than 10 percent GPU memory headroom')
    record = {'rank': rank, 'host': socket.gethostname(), 'passed': True,
              'image_ids': [t['image_id'] for t in targets], 'batch_size': len(images),
              'normalization_max_abs_error': normalization_error,
              'first_step': first, 'uninterrupted_step': uninterrupted, 'replayed_step': replay,
              'resume_parameter_max_abs_error': max_error, 'frozen_parameters_unchanged': True,
              'frozen_parameter_sha256': frozen_hash,
              'ranks_synchronized': True, 'peak_allocated_bytes': allocated,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'gpu_total_bytes': total, 'seconds': time.monotonic() - started,
              'test_labels_read': False, 'protocol_sha256': fingerprint}
    atomic_json(output / f'rank{rank}.json', record)
    records = [None] * world
    dist.all_gather_object(records, record)
    if rank == 0:
        atomic_json(output / 'passed.json', {'passed': True, 'world_size': world,
                    'protocol_sha256': fingerprint, 'qualification_only_not_accuracy': True,
                    'selection': '16 train images ranked by unique annotated humans then edge count',
                    'ranks': records})
        print(json.dumps({'qualification_passed': True, 'path': str(output / 'passed.json')}), flush=True)
    dist.barrier()
