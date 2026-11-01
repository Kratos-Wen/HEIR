"""CoRISP training and cache generation on V-COCO."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
HARNESS_ROOT = Path(os.environ.get('PVIC_ROOT', WORKSPACE / 'vendor' / 'pvic')).expanduser().resolve()
SLHOI_ROOT = Path(os.environ.get('SLHOI_ROOT', WORKSPACE / 'vendor' / 'slhoi'))
if not (HARNESS_ROOT / 'main.py').is_file():
    raise FileNotFoundError(f'The V-COCO PViC harness was not found under {HARNESS_ROOT}.')
for path in (WORKSPACE / 'src', HERE, HARNESS_ROOT, HARNESS_ROOT / 'pocket', HARNESS_ROOT / 'h_detr' / 'models' / 'ops', SLHOI_ROOT):
    sys.path.insert(0, str(path))
from configs import advanced_detector_args
from hdetr_corisp_vcoco import build_vcoco_model
from pocket.core import DistributedLearningEngine
from utils import CustomisedDLE, DataFactory, custom_collate
from corisp import VCOCORoleSpace, load_vcoco_null_role_index
from vcoco_null_dataset import VCOCONullAwareDataFactory, include_official_vcoco_images


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _relocate_optimizer_state(optimizer, rank):
    device = torch.device('cuda', rank)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class VCOCOTrainingEngine(CustomisedDLE):
    """Train without repeatedly inspecting the official V-COCO test split."""

    def __init__(self, net, train_dataloader, test_dataloader, config, device: int):
        DistributedLearningEngine.__init__(self, net, None, train_dataloader, device=device, print_interval=config.print_interval, cache_dir=config.output_dir, find_unused_parameters=True)
        self.config = config
        self.max_norm = config.clip_max_norm
        self.test_dataloader = test_dataloader
        self.grad_accum_steps = int(config.grad_accum_steps)

    def _on_start(self) -> None:
        self.best_perf = float('-inf')

    def _on_end(self) -> None:
        return None

    def _assert_finite_gradients(self) -> None:
        invalid: list[str] = []
        for name, parameter in self._state.net.named_parameters():
            gradient = parameter.grad
            if gradient is None or torch.isfinite(gradient).all():
                continue
            invalid_count = int((~torch.isfinite(gradient)).sum().item())
            invalid.append(f'{name}({invalid_count}/{gradient.numel()})')
            if len(invalid) == 16:
                break
        if invalid:
            raise FloatingPointError(f'Non-finite V-COCO gradients on rank {self._rank}: ' + ', '.join(invalid))

    def _on_each_iteration(self) -> None:
        """BF16 training with deterministic microbatch accumulation."""
        epoch_length = len(self._train_loader)
        position = (int(self._state.iteration) - 1) % epoch_length
        window_start = position // self.grad_accum_steps * self.grad_accum_steps
        window_size = min(self.grad_accum_steps, epoch_length - window_start)
        at_window_start = position == window_start
        at_window_end = position + 1 == window_start + window_size
        if at_window_start:
            self._state.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss_dict = self._state.net(*self._state.inputs, targets=self._state.targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite V-COCO loss on rank {self._rank}: {loss_dict}')
        (loss / float(window_size)).backward()
        if at_window_end:
            self._assert_finite_gradients()
            if self.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm, error_if_nonfinite=True)
            self._state.optimizer.step()
        self._state.loss = loss.detach()

    def _print_statistics(self) -> None:
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size
        if self._rank == 0:
            num_iter = len(self._train_loader)
            current = self._state.iteration - num_iter * (self._state.epoch - 1)
            print('Epoch [{}/{}], Iter. [{}/{}], Loss: {:.4f}, Time[Data/Iter.]: [{:.2f}s/{:.2f}s]'.format(self._state.epoch, self.config.epochs, current, num_iter, running_loss, t_data, t_iter))
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_end_epoch(self) -> None:
        """Save every stage while keeping the official test sealed until evaluation."""
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        rng_states = [None] * dist.get_world_size()
        dist.all_gather_object(rng_states, rng)
        if self._rank == 0:
            epoch = int(self._state.epoch)
            checkpoint = {'iteration': self._state.iteration, 'epoch': epoch, 'performance': None, 'selection_rule': 'fixed_final_epoch_no_test_selection', 'model_state_dict': self._state.net.module.state_dict(), 'optim_state_dict': self._state.optimizer.state_dict(), 'scaler_state_dict': self._state.scaler.state_dict(), 'rng_states': rng_states, 'world_size': self.config.world_size, 'scheduler_timing': 'saved_before_step_restore_steps_once'}
            if self._state.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
            output_dir = Path(self._cache_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            def atomic_save(path):
                temporary = path.with_suffix('.tmp')
                torch.save(checkpoint, temporary)
                temporary.replace(path)
            atomic_save(output_dir / 'latest.pth')
            if epoch in set(self.config.checkpoint_epochs):
                atomic_save(output_dir / f'checkpoint_epoch_{epoch:03d}.pth')
            if epoch == int(self.config.epochs):
                atomic_save(output_dir / 'final.pth')
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[advanced_detector_args()])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--box-score-thresh', default=0.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--vcoco-role-loss-weight', default=1.0, type=float)
    parser.add_argument('--vcoco-train-vsrl-json', default='')
    parser.add_argument('--vcoco-coco-json', default='')
    parser.add_argument('--vcoco-eval-split-ids', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--checkpoint-epochs', nargs='*', type=int, default=())
    parser.add_argument('--port', default='29561')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world-size', default=2, type=int)
    parser.add_argument('--grad-accum-steps', default=1, type=int)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--adam-beta1', default=0.9, type=float)
    parser.add_argument('--adam-beta2', default=0.999, type=float)
    parser.add_argument('--adam-eps', default=1e-08, type=float)
    return parser

def _validate_protocol(args: argparse.Namespace) -> None:
    if args.batch_size % args.world_size:
        raise ValueError('Global batch size must be divisible by world size.')
    if not args.vcoco_train_vsrl_json or not args.vcoco_coco_json:
        raise ValueError('V-COCO requires the declared training VSRL and COCO JSON files.')
    if args.world_size not in (1, 2, 4, 8):
        raise ValueError('CoRISP supports 1, 2, 4, or 8 data-parallel workers.')
    if args.grad_accum_steps <= 0:
        raise ValueError('grad_accum_steps must be positive.')
    if args.epochs != 30:
        raise ValueError('The published V-COCO training schedule uses 30 epochs.')
    checkpoint_epochs = list(args.checkpoint_epochs)
    if checkpoint_epochs != sorted(set(checkpoint_epochs)):
        raise ValueError('checkpoint_epochs must be unique and strictly increasing.')
    if any(epoch <= 0 or epoch > args.epochs for epoch in checkpoint_epochs):
        raise ValueError('checkpoint_epochs must fall inside the training budget.')
    expected = {'backbone': 'swin_large', 'drop_path_rate': 0.5, 'hidden_dim': 256, 'enc_layers': 6, 'dec_layers': 6,
        'dim_feedforward': 2048, 'dropout': 0.0, 'nheads': 8, 'num_feature_levels': 4, 'num_queries_one2one': 900,
        'num_queries_one2many': 1500, 'with_box_refine': True, 'two_stage': True, 'mixed_selection': True,
        'look_forward_twice': True, 'masks': False, 'aux_loss': True, 'epochs': 30, 'lr_head': 0.0001, 'lr_backbone': 0.0,
        'lr_drop': 20, 'lr_drop_factor': 0.2, 'weight_decay': 0.0001, 'clip_max_norm': 0.1, 'adam_beta1': 0.9,
        'adam_beta2': 0.999, 'adam_eps': 1e-08, 'repr_dim': 384, 'alpha': 0.5, 'gamma': 0.1, 'box_score_thresh': 0.05,
        'min_instances': 3, 'max_instances': 15, 'vcoco_role_loss_weight': 1.0, 'dataset': 'vcoco',
        'partitions': ['trainval', 'test'], 'topk': 100, 'focal_alpha': 0.25, 'num_workers': 2, 'print_interval': 100,
        'batch_size': 1 if args.cache else 8, 'grad_accum_steps': 1 if args.cache else 2}
    mismatches = {key: {'expected': value, 'actual': getattr(args, key)} for key, value in expected.items() if getattr(args, key) != value}
    if mismatches:
        raise ValueError(f'V-COCO configuration mismatch: {mismatches}')
    if not args.cache and not args.use_checkpoint:
        raise ValueError('Swin-L training requires --use-checkpoint.')
    if args.cache and not args.resume:
        raise ValueError('Cache generation requires --resume with a trained checkpoint.')

def _make_loaders(rank: int, args: argparse.Namespace):
    trainset = DataFactory('vcoco', args.partitions[0], args.data_root)
    role_space = VCOCORoleSpace.from_role_classes(trainset.dataset.actions, trainset.dataset.num_instances)
    null_index = load_vcoco_null_role_index(args.vcoco_train_vsrl_json, args.vcoco_coco_json, role_space)
    trainset = VCOCONullAwareDataFactory(trainset, null_index)
    per_device = args.batch_size // args.world_size
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
    train_loader = DataLoader(trainset, collate_fn=custom_collate, batch_size=per_device, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    if not args.cache:
        return (trainset, train_loader, None)
    testset = DataFactory('vcoco', args.partitions[1], args.data_root)
    if not args.vcoco_eval_split_ids:
        raise ValueError('V-COCO requires the declared evaluation image-id split.')
    official_test_ids = np.atleast_1d(np.loadtxt(args.vcoco_eval_split_ids, dtype=np.int64)).tolist()
    include_official_vcoco_images(testset, official_test_ids)
    test_sampler = DistributedSampler(testset, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=True)
    test_loader = DataLoader(testset, collate_fn=custom_collate, batch_size=per_device, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    return (trainset, train_loader, test_loader)

def _worker(rank: int, args: argparse.Namespace, local_rank: int | None=None) -> None:
    local_rank = rank if local_rank is None else local_rank
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    try:
        torch.cuda.set_device(local_rank)
        seed = args.seed + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        trainset, train_loader, test_loader = _make_loaders(rank, args)
        model = build_vcoco_model(args, trainset.dataset)
        resume_checkpoint = None
        if args.resume:
            resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(resume_checkpoint['model_state_dict'], strict=True)
        model.freeze_detector()
        engine = VCOCOTrainingEngine(model, train_loader, test_loader, args, device=local_rank)
        if rank == 0:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            resolved = vars(args).copy()
            resolved.update({'detector_checkpoint_sha256': _sha256(args.pretrained), 'trainable_parameter_count': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'frozen_parameter_count': sum((p.numel() for p in model.parameters() if not p.requires_grad))})
            (output / 'resolved_training_config.json').write_text(json.dumps(resolved, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            (output / 'vcoco_role_space.json').write_text(json.dumps(model.role_space.as_dict(), indent=2) + '\n', encoding='utf-8')
            (output / 'vcoco_null_role_summary.json').write_text(json.dumps(trainset.null_index.summary(model.role_space), indent=2) + '\n', encoding='utf-8')
        if args.cache:
            engine.cache_vcoco(test_loader, args.output_dir)
            return
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{'params': parameters, 'lr': args.lr_head}], lr=args.lr_head, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_drop_factor)
        start_epoch = 0
        start_iteration = 0
        if resume_checkpoint is not None:
            required = {'optim_state_dict', 'scheduler_state_dict', 'epoch', 'iteration', 'rng_states'}
            missing = required.difference(resume_checkpoint)
            if missing:
                raise RuntimeError(f'True resume requires optimizer/scheduler/progress state; missing {sorted(missing)}.')
            optimizer.load_state_dict(resume_checkpoint['optim_state_dict'])
            _relocate_optimizer_state(optimizer, local_rank)
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            scheduler.step()
            if len(resume_checkpoint['rng_states']) != args.world_size:
                raise ValueError('Resume requires unchanged worker topology')
            rng = resume_checkpoint['rng_states'][rank]
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'])
            start_epoch = int(resume_checkpoint['epoch'])
            start_iteration = int(resume_checkpoint['iteration'])
            if 'scaler_state_dict' in resume_checkpoint:
                engine._state.scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
        if start_epoch >= args.epochs:
            raise ValueError(f'Checkpoint epoch {start_epoch} reached the {args.epochs}-epoch budget.')
        engine.update_state_key(optimizer=optimizer, lr_scheduler=scheduler, epoch=start_epoch, iteration=start_iteration)
        engine(args.epochs - start_epoch)
    finally:
        dist.destroy_process_group()

def main() -> None:
    args = _build_parser().parse_args()
    _validate_protocol(args)
    if args.validate_only:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return
    print(args)
    os.environ['WANDB_MODE'] = 'disabled'
    distributed_env = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')
    present = [name in os.environ for name in distributed_env]
    if any(present) and (not all(present)):
        missing = [name for name, exists in zip(distributed_env, present) if not exists]
        raise RuntimeError(f'Incomplete torchrun environment; missing {missing}.')
    if all(present):
        if int(os.environ['WORLD_SIZE']) != args.world_size:
            raise ValueError('torchrun WORLD_SIZE does not match --world-size.')
        _worker(int(os.environ['RANK']), args, local_rank=int(os.environ['LOCAL_RANK']))
        return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    mp.spawn(_worker, nprocs=args.world_size, args=(args,))
if __name__ == '__main__':
    main()
