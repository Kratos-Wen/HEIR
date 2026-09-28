"""Train CoRISP on HEIR with frozen visual encoders."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import random
from types import MethodType

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .environment import restore_execution, verify_core
from .checkpoint import load_adaptation, move, restore_rng, save
from heir_protocol.compatibility import Compatibility
from heir_training.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'detector', 'prototypes', 'compatibility', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--ablation', choices=('full', 'no_arity', 'no_relations', 'no_role_feedback'), default='full')
    parser.add_argument('--compile-dp', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or args.workers < 0:
        raise ValueError('Invalid epoch or worker count')
    from .data import HEIREvents, collate
    from .model import build
    from corisp_heir_bucket.packed import forward
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA and a compute allocation')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    try:
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        torch.manual_seed(42 + rank)
        torch.backends.cudnn.benchmark = False
        execution = dict(dp_backend='adjoint', compiled_dp=args.compile_dp,
                         metadata_cache=True, selective_recomputation=False)
        restore_execution(execution)
        sources = verify_core()
        data = HEIREvents(args.data)
        support = json.loads(args.compatibility.read_text())
        model = build(data, args.detector, args.prototypes, support, args.ablation).to(device)
        model.heir_packed_chunk = 256
        model.forward = MethodType(forward, model)
        protocol = dict(
            schema='corisp_heir_training_v1', model='CoRISP', ablation=args.ablation,
            seed=42, epochs=args.epochs, world_size=world, per_gpu_batch=1,
            accumulation=2, global_batch=2 * world, workers=args.workers,
            optimizer='AdamW', lr=1e-4, weight_decay=1e-4, lr_drop=20, lr_gamma=.2,
            clip_grad_norm=.1, alpha=.5, gamma=.1, execution=execution,
            precision='BF16 visual computations; FP32 structured potentials and DP',
            compatibility_binding=Compatibility(support).binding(),
            inputs={'train.json': sha256(data.root / 'annotations/train.json'),
                    'vocabulary.json': sha256(data.root / 'vocabulary.json'),
                    'detector': sha256(args.detector), 'prototypes': sha256(args.prototypes)},
            sources=sources)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=.2)
        start = 0
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['protocol'] != protocol or len(state['rng_states']) != world:
                raise ValueError('Resume requires the same inputs, configuration and world size')
            load_adaptation(model, state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            start = state['epoch']
            restore_rng(state['rng_states'][rank])
            del state
        # Check this on every rank before entering a barrier.
        if not args.resume and args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Use an empty output directory')
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / 'config.json').write_text(json.dumps(protocol, indent=2) + '\n')
        dist.barrier()
        ddp = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)
        sampler = DistributedSampler(data, shuffle=True, seed=42, drop_last=True)
        for epoch in range(start, args.epochs):
            sampler.set_epoch(epoch)
            generator = torch.Generator().manual_seed(42 + epoch * world + rank)
            loader = DataLoader(data, batch_size=1, sampler=sampler, num_workers=args.workers,
                                pin_memory=True, collate_fn=collate, generator=generator)
            steps = (len(loader) // 2) * 2
            if steps == 0:
                raise ValueError('No complete accumulation step in this training split')
            ddp.train()
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                if step >= steps:
                    break
                synchronize = step % 2 == 1
                with (nullcontext() if synchronize else ddp.no_sync()):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = ddp(*move(batch, device))
                    if not torch.isfinite(result['loss']):
                        raise FloatingPointError('Nonfinite set loss')
                    (result['loss'] / 2).backward()
                if synchronize:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            save(args.output / f'epoch_{epoch + 1:03}.pth', model, optimizer, scheduler,
                 epoch + 1, protocol)
            if rank == 0:
                print(f'epoch={epoch + 1} loss={float(result["loss"]):.6f}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
