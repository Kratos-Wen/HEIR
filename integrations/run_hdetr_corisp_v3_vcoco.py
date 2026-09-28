"""Training and official-role evaluation for complete CoRISP on V-COCO."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CORISP_SRC = WORKSPACE / 'src'
DEFAULT_HARNESS = WORKSPACE / 'vendor' / 'pvic'
HARNESS_ROOT = Path(os.environ.get('PVIC_ROOT', DEFAULT_HARNESS)).expanduser().resolve()
HDETR_OPS = HARNESS_ROOT / 'h_detr' / 'models' / 'ops'
POCKET_ROOT = HARNESS_ROOT / 'pocket'
CLIP_CODE_ROOT = Path(os.environ.get('CLIP_CODE_ROOT', '')).expanduser()
SLHOI_ROOT = Path(os.environ.get('SLHOI_ROOT', WORKSPACE / 'vendor' / 'slhoi'))
if not (HARNESS_ROOT / 'main.py').is_file():
    raise FileNotFoundError(f'The V-COCO PViC harness was not found under {HARNESS_ROOT}.')
paths = [CORISP_SRC, HERE, HARNESS_ROOT, POCKET_ROOT, HDETR_OPS]
if SLHOI_ROOT.is_dir():
    paths.append(SLHOI_ROOT)
if str(CLIP_CODE_ROOT) and CLIP_CODE_ROOT.is_dir():
    paths.insert(0, CLIP_CODE_ROOT)
for path in paths:
    sys.path.insert(0, str(path))
from configs import advanced_detector_args
from hdetr_corisp_role_arity_event_field_vcoco import build_hdetr_corisp_role_arity_event_field_vcoco
from pocket.core import DistributedLearningEngine
from pocket.ops import relocate_to_cuda
from checkpoint_io import _model_state, _relocate_optimizer_state, _sha256
from utils import CustomisedDLE, DataFactory, custom_collate
from corisp import VCOCORoleSpace, load_vcoco_null_role_index
from vcoco_null_dataset import VCOCONullAwareDataFactory, include_official_vcoco_images

def _variant() -> str:
    return 'hdetr_corisp_role_arity_event_field'

def _architecture_id() -> str:
    return 'corisp_joint_v11_role_arity_event_field_v1_vcoco'
_pvic_test_vcoco = CustomisedDLE.test_vcoco

def _test_vcoco_with_artifact(self):
    """Persist PViC's 24-class diagnostic AP; this is not official S1/S2."""
    ap = _pvic_test_vcoco(self)
    if self._rank == 0:
        dataset = self.test_dataloader.dataset.dataset
        epoch = getattr(self._state, 'epoch', 0)
        payload = {'schema': 'corisp_vcoco_diagnostic_metrics_v1', 'metric_scope': 'diagnostic_nonofficial_24_role_class_pair_ap', 'official_result_required': True, 'architecture_id': _architecture_id(), 'variant': 'hdetr_corisp_role_arity_event_field', 'protocol': self.config.protocol, 'epoch': 0 if epoch is None else int(epoch), 'mean_ap': float(ap.mean()), 'per_class_ap': [float(value) for value in ap], 'class_names': list(dataset.actions), 'test_images': len(dataset), 'unix_time': time.time()}
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'eval_diagnostic_latest.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        with (output_dir / 'eval_diagnostic_history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, separators=(',', ':')) + '\n')
    return ap
CustomisedDLE.test_vcoco = _test_vcoco_with_artifact

class _VCOCOTrainingDLE(CustomisedDLE):
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
            selection_rule = 'fixed_final_epoch_no_test_selection'
            if self.config.convergence_probe:
                selection_rule = 'validation_curve_only_no_official_test_selection' if self.config.vcoco_split_protocol == 'official_train_val_dev' else 'test_exposed_optimization_diagnostic_not_for_selection'
            checkpoint = {'iteration': self._state.iteration, 'epoch': epoch, 'performance': None, 'selection_rule': selection_rule, 'convergence_probe': bool(self.config.convergence_probe), 'model_state_dict': self._state.net.module.state_dict(), 'optim_state_dict': self._state.optimizer.state_dict(), 'scaler_state_dict': self._state.scaler.state_dict(), 'rng_states': rng_states, 'world_size': self.config.world_size, 'scheduler_timing': 'saved_before_step_restore_steps_once'}
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
    parser.add_argument('--raw-lambda', default=1.7, type=float)
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--box-score-thresh', default=0.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--vcoco-role-loss-weight', default=1.0, type=float)
    parser.add_argument('--vcoco-train-vsrl-json', '--vcoco-trainval-vsrl-json', dest='vcoco_train_vsrl_json', default='')
    parser.add_argument('--vcoco-coco-json', default='')
    parser.add_argument('--vcoco-eval-split-ids', '--vcoco-test-split-ids', dest='vcoco_eval_split_ids', default='')
    parser.add_argument('--vcoco-split-protocol', choices=('official_trainval_test', 'official_train_val_dev'), default='official_trainval_test')
    parser.add_argument('--resume', default='')
    parser.add_argument('--checkpoint-epochs', nargs='*', type=int, default=())
    parser.add_argument('--port', default='29561')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world-size', default=2, type=int)
    parser.add_argument('--grad-accum-steps', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--protocol', default='vcoco_role_arity_event_field_matched', choices=('vcoco_role_arity_event_field_matched',))
    parser.add_argument('--adam-beta1', default=0.9, type=float)
    parser.add_argument('--adam-beta2', default=0.999, type=float)
    parser.add_argument('--adam-eps', default=1e-08, type=float)
    return parser

def _validate_protocol(args: argparse.Namespace) -> None:
    if sum((args.eval, args.cache, args.smoke)) > 1:
        raise ValueError('--eval, --cache, and --smoke are mutually exclusive.')
    if args.batch_size % args.world_size:
        raise ValueError('Global batch size must be divisible by world size.')
    evaluation = args.eval or args.cache
    if args.smoke:
        if args.world_size not in (1, 2, 4, 8) or args.batch_size != args.world_size:
            raise ValueError('Smoke tests require one sample per rank and support 1, 2, 4, or 8 ranks.')
    semantic_event_set = False
    agent_role_field = True
    event_field = True
    expected_event_protocol = 'vcoco_role_arity_event_field_matched'
    if event_field and args.protocol != expected_event_protocol:
        raise ValueError(f"{'hdetr_corisp_role_arity_event_field'} requires protocol={expected_event_protocol}.")
    if event_field:
        if not args.vcoco_train_vsrl_json or not args.vcoco_coco_json:
            raise ValueError('CoRISP event models require the declared training VSRL and COCO JSON files.')
        if args.world_size not in (1, 2, 4, 8):
            raise ValueError('CoRISP event models support 1, 2, 4, or 8 data-parallel workers.')
    if args.grad_accum_steps <= 0:
        raise ValueError('grad_accum_steps must be positive.')
    if args.epochs != 30:
        raise ValueError('The published V-COCO training schedule uses 30 epochs.')
    checkpoint_epochs = list(args.checkpoint_epochs)
    if checkpoint_epochs != sorted(set(checkpoint_epochs)):
        raise ValueError('checkpoint_epochs must be unique and strictly increasing.')
    if any((epoch <= 0 or epoch > args.epochs for epoch in checkpoint_epochs)):
        raise ValueError('checkpoint_epochs must fall inside the training budget.')
    development_split = args.vcoco_split_protocol == 'official_train_val_dev'
    expected_partitions = ['train', 'val'] if development_split else ['trainval', 'test']
    if development_split and (args.eval or args.cache) and (args.world_size != 1):
        raise ValueError('Development validation cache/evaluation must use one worker so no validation image is dropped.')
    expected = {'backbone': 'swin_large', 'drop_path_rate': 0.5, 'hidden_dim': 256, 'enc_layers': 6, 'dec_layers': 6, 'dim_feedforward': 2048, 'dropout': 0.0, 'nheads': 8, 'num_feature_levels': 4, 'num_queries_one2one': 900, 'num_queries_one2many': 1500, 'with_box_refine': True, 'two_stage': True, 'mixed_selection': True, 'look_forward_twice': True, 'masks': False, 'aux_loss': True, 'epochs': 30, 'lr_head': 0.0001, 'lr_backbone': 0.0, 'lr_drop': 20, 'lr_drop_factor': 0.2, 'weight_decay': 0.0001, 'clip_max_norm': 0.1, 'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_eps': 1e-08, 'repr_dim': 384, 'alpha': 0.5, 'gamma': 0.1, 'raw_lambda': 1.0 if agent_role_field else 1.7, 'box_score_thresh': 0.05, 'min_instances': 3, 'max_instances': 15, 'vcoco_role_loss_weight': 1.0, 'dataset': 'vcoco', 'partitions': expected_partitions, 'topk': 100, 'focal_alpha': 0.25, 'num_workers': 2, 'print_interval': 100, 'batch_size': args.batch_size if args.smoke else 1 if evaluation else args.world_size if development_split else 8 if semantic_event_set or agent_role_field else 16, 'grad_accum_steps': 1 if args.smoke or evaluation or (not (semantic_event_set or agent_role_field)) else 16 // args.world_size if development_split else 2, 'world_size': args.world_size if args.smoke or event_field else 1 if evaluation else 2}
    mismatches = {key: {'expected': value, 'actual': getattr(args, key)} for key, value in expected.items() if getattr(args, key) != value}
    if mismatches:
        raise ValueError(f'V-COCO configuration mismatch: {mismatches}')
    if not evaluation and (not args.use_checkpoint):
        raise ValueError('Swin-L training requires --use-checkpoint.')
    if (args.eval or args.cache) and (not args.resume):
        raise ValueError('Evaluation/cache requires --resume with a trained checkpoint.')

def _make_loaders(rank: int, args: argparse.Namespace):
    trainset = DataFactory('vcoco', args.partitions[0], args.data_root)
    role_space = VCOCORoleSpace.from_role_classes(trainset.dataset.actions, trainset.dataset.num_instances)
    null_index = load_vcoco_null_role_index(args.vcoco_train_vsrl_json, args.vcoco_coco_json, role_space)
    trainset = VCOCONullAwareDataFactory(trainset, null_index)
    per_device = args.batch_size // args.world_size
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
    train_loader = DataLoader(trainset, collate_fn=custom_collate, batch_size=per_device, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    if not (args.eval or args.cache):
        return (trainset, train_loader, None)
    testset = DataFactory('vcoco', args.partitions[1], args.data_root)
    if not args.vcoco_eval_split_ids:
        raise ValueError('V-COCO requires the declared evaluation image-id split.')
    official_test_ids = np.atleast_1d(np.loadtxt(args.vcoco_eval_split_ids, dtype=np.int64)).tolist()
    include_official_vcoco_images(testset, official_test_ids)
    train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
    test_sampler = DistributedSampler(testset, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=True)
    per_device = args.batch_size // args.world_size
    train_loader = DataLoader(trainset, collate_fn=custom_collate, batch_size=per_device, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
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
        model = build_hdetr_corisp_role_arity_event_field_vcoco(args, trainset.dataset)
        resume_checkpoint = None
        if args.resume:
            resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model.load_state_dict(_model_state(resume_checkpoint), strict=True)
        model.freeze_detector()
        engine = _VCOCOTrainingDLE(model, train_loader, test_loader, args, device=local_rank)
        if rank == 0:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            resolved = vars(args).copy()
            resolved.update({'schema': 'corisp_vcoco_resolved_training_config_v1', 'architecture_id': _architecture_id(), 'variant': 'hdetr_corisp_role_arity_event_field', 'detector_checkpoint_sha256': _sha256(args.pretrained), 'trainable_parameter_count': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'frozen_parameter_count': sum((p.numel() for p in model.parameters() if not p.requires_grad))})
            (output / 'resolved_training_config.json').write_text(json.dumps(resolved, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            (output / 'vcoco_role_space.json').write_text(json.dumps(model.role_space.as_dict(), indent=2) + '\n', encoding='utf-8')
            if isinstance(trainset, VCOCONullAwareDataFactory):
                (output / 'vcoco_null_role_summary.json').write_text(json.dumps(trainset.null_index.summary(model.role_space), indent=2) + '\n', encoding='utf-8')
        if args.cache:
            engine.cache_vcoco(test_loader, args.output_dir)
            return
        if args.eval:
            engine.test_vcoco()
            return
        if args.smoke:
            engine._state.net.train()
            parameters = [parameter for parameter in engine._state.net.parameters() if parameter.requires_grad]
            smoke_optimizer = torch.optim.AdamW([{'params': parameters, 'lr': args.lr_head}], lr=args.lr_head, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
            loader_iterator = iter(train_loader)
            smoke_steps = 2
            smoke_gradient_activity: dict[str, bool] = {}
            stress_shape = (800, 1333)
            torch.cuda.reset_peak_memory_stats(local_rank)
            for _ in range(smoke_steps):
                smoke_optimizer.zero_grad(set_to_none=True)
                images, targets = next(loader_iterator)
                images = relocate_to_cuda(images)
                targets = relocate_to_cuda(targets)
                images = [F.interpolate(image[None], size=stress_shape, mode='bilinear', align_corners=False, antialias=True)[0] for image in images]
                targets = [{**target, 'size': target['size'].new_tensor(stress_shape)} for target in targets]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    losses = engine._state.net(images, targets=targets)
                    total = sum(losses.values())
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Non-finite V-COCO smoke loss: {losses}')
                total.backward()
                invalid_gradients = []
                for name, parameter in engine._state.net.named_parameters():
                    if parameter.grad is not None and (not torch.isfinite(parameter.grad).all()):
                        invalid_gradients.append(name)
                if invalid_gradients:
                    raise FloatingPointError('Non-finite V-COCO smoke gradients: ' + ', '.join(invalid_gradients[:16]))
                if False and _ == smoke_steps - 1:
                    network = getattr(engine._state.net, 'module', engine._state.net)
                    expected_binding_gradients = {'pair_query': network.role_field.pair_query_projection[1].weight, 'semantic_query': network.role_field.semantic_query_projection[1].weight, 'dense_key': network.role_field.dense_key_projection.weight, 'dense_value': network.role_field.dense_value_projection.weight, 'state_energy': network.role_field.retrieved_state_energy.weight}
                    inactive = [name for name, parameter in expected_binding_gradients.items() if parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0]
                    if inactive:
                        raise RuntimeError('Pair-frame smoke found inactive binding gradients: ' + ', '.join(inactive))
                    leaked_frozen_gradients = [name for name, parameter in network.semantic_backbone.named_parameters() if parameter.grad is not None]
                    if leaked_frozen_gradients:
                        raise RuntimeError('Frozen DINO parameters received gradients: ' + ', '.join(leaked_frozen_gradients[:16]))
                network = getattr(engine._state.net, 'module', engine._state.net)
                field = network.role_filler_event_field
                expected_field_gradients = {'semantic_host': network.role_field.host_semantic_projection[1].weight}
                expected_field_gradients.update({'cardinality': field.cardinality_head.weight, 'relational_gate': field.refinement_log_scale, 'role_state_query': field.query_projection[1].weight, 'role_state_value': field.value_projection[1].weight, 'arity_gate': field.arity_log_scale, 'arity_role': field.arity_role_projection[1].weight, 'arity_count': field.arity_count_embeddings, 'arity_set': field.arity_set_projection[1].weight})
                for name, parameter in expected_field_gradients.items():
                    active = parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
                    smoke_gradient_activity[name] = smoke_gradient_activity.get(name, False) or active
                if _ == smoke_steps - 1:
                    inactive = [name for name in expected_field_gradients if not smoke_gradient_activity.get(name, False)]
                    if inactive:
                        raise RuntimeError('Structured role event-field smoke found gradients inactive across every optimizer step: ' + ', '.join(inactive))
                leaked_frozen_gradients = [name for name, parameter in network.semantic_backbone.named_parameters() if parameter.grad is not None]
                if leaked_frozen_gradients:
                    raise RuntimeError('Frozen DINO parameters received gradients: ' + ', '.join(leaked_frozen_gradients[:16]))
                if False and _ == smoke_steps - 1:
                    network = getattr(engine._state.net, 'module', engine._state.net)
                    ablation = network.ablation
                    expected_role_set_gradients = {'semantic_host': network.role_field.host_semantic_projection[1].weight, 'agent_alignment': network.role_field.agent_query_projection[1].weight}
                    if ablation.learn_cardinality:
                        expected_role_set_gradients['cardinality'] = network.grounded_role_set.cardinality_head.weight
                    if ablation.use_line_graph:
                        expected_role_set_gradients.update({'line_graph_attention': network.role_field.line_graph.attention.in_proj_weight, 'line_graph_relation_gate': network.role_field.line_graph.relation_gate[1].weight, 'line_graph_state': network.role_field.line_graph.state_embeddings})
                    inactive = [name for name, parameter in expected_role_set_gradients.items() if parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0]
                    if inactive:
                        raise RuntimeError('Grounded role-set smoke found inactive gradients: ' + ', '.join(inactive))
                    leaked_frozen_gradients = [name for name, parameter in network.semantic_backbone.named_parameters() if parameter.grad is not None]
                    if leaked_frozen_gradients:
                        raise RuntimeError('Frozen DINO parameters received gradients: ' + ', '.join(leaked_frozen_gradients[:16]))
                if False and _ == smoke_steps - 1:
                    network = getattr(engine._state.net, 'module', engine._state.net)
                    graph = network.role_field.line_graph
                    expected_graph_gradients = {'attention': graph.attention.in_proj_weight, 'relation_gate': graph.relation_gate[1].weight, 'state_embedding': graph.state_embeddings, 'state_correction': graph.state_correction[1].weight}
                    inactive = [name for name, parameter in expected_graph_gradients.items() if parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0]
                    if inactive:
                        raise RuntimeError('Line-graph smoke found inactive gradients: ' + ', '.join(inactive))
                    leaked_frozen_gradients = [name for name, parameter in network.semantic_backbone.named_parameters() if parameter.grad is not None]
                    if leaked_frozen_gradients:
                        raise RuntimeError('Frozen DINO parameters received gradients: ' + ', '.join(leaked_frozen_gradients[:16]))
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(engine._state.net.parameters(), args.clip_max_norm, error_if_nonfinite=True)
                smoke_optimizer.step()
            torch.cuda.synchronize(local_rank)
            peak = torch.tensor(float(torch.cuda.max_memory_allocated(local_rank)), device=local_rank, dtype=torch.float64)
            capacity = torch.tensor(float(torch.cuda.get_device_properties(local_rank).total_memory), device=local_rank, dtype=torch.float64)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            dist.all_reduce(capacity, op=dist.ReduceOp.MIN)
            headroom = capacity - peak
            required_headroom = 4 * 2 ** 30
            if headroom.item() < required_headroom:
                raise RuntimeError(f'CUDA stress smoke left less than 4 GiB headroom: peak={peak.item() / 2 ** 20:.1f} MiB, minimum_capacity={capacity.item() / 2 ** 20:.1f} MiB.')
            if rank == 0:
                artifact = {'schema': 'corisp_vcoco_cuda_smoke_v2', 'variant': 'hdetr_corisp_role_arity_event_field', 'resume_checkpoint': args.resume or None, 'steps': smoke_steps, 'world_size': args.world_size, 'per_rank_batch_size': args.batch_size // args.world_size, 'stress_shape': list(stress_shape), 'optimizer_step_included': True, 'losses': {name: float(value.detach()) for name, value in losses.items()}, 'loss_dtypes': {name: str(value.dtype) for name, value in losses.items()}, 'total_loss': float(total.detach()), 'max_memory_mib_all_ranks': float(peak.item()) / 2 ** 20, 'minimum_capacity_mib': float(capacity.item()) / 2 ** 20, 'minimum_headroom_mib': float(headroom.item()) / 2 ** 20, 'required_headroom_mib': float(required_headroom) / 2 ** 20, 'gradient_activity_across_steps': smoke_gradient_activity, 'status': 'passed'}
                (Path(args.output_dir) / 'cuda_smoke.json').write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8')
                print(json.dumps(artifact, indent=2))
            dist.barrier()
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
