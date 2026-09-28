"""Real-data full-model qualification, never an accuracy result."""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .runtime import build_runtime, load_detr, optimizer, train_dataset, batch_to_device, total_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl')
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)
    model, criterion, vlm, config, vis, txt = build_runtime(device)
    loading = load_detr(model)
    opt, groups = optimizer(model, config)
    data = train_dataset(vis, txt, config)
    from util.misc import collate_fn
    wrapped = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    records = []
    for empty in (False, True):
        batch = collate_fn([data[rank * 4 + i] for i in range(4)])
        samples, targets, clip = batch_to_device(batch, device)
        if empty:
            for target in targets:
                for key in ('obj_labels', 'verb_labels', 'hoi_labels', 'sub_boxes', 'obj_boxes'):
                    target[key] = target[key][:0]
        opt.zero_grad(set_to_none=True)
        output = wrapped(samples, vlm, clip_input=clip)
        loss, losses = total_loss(criterion, output, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        if not torch.isfinite(norm) or any(p.grad is not None for p in vlm.parameters()):
            raise ValueError('Invalid gradient or unfrozen CLIP')
        opt.step()
        record = {'rank': rank, 'empty_targets': empty, 'loss': float(loss), 'gradient_norm': float(norm),
                  'loss_terms': {k: float(v) for k, v in losses.items()},
                  'peak_mib': torch.cuda.max_memory_allocated() / 1024**2,
                  'filenames': [t['filename'] for t in targets]}
        print(json.dumps(record), flush=True)
        records.append(record)
    all_records = [records]
    if world > 1:
        all_records = [None] * world
        dist.all_gather_object(all_records, records)
    if rank == 0:
        from pathlib import Path
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps({'status': 'full_model_real_train_backward_passed',
                        'world_size': world, 'batch_per_gpu': 4, 'records': all_records,
                        'detr_loading': loading, 'optimizer_groups': groups,
                        'accuracy_verified': False}, indent=2) + '\n')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
