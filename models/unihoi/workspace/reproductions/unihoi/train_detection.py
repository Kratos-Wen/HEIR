"""Fixed-budget supervised adaptation; NOT the full UniHOI cycle recipe."""

import argparse
from datetime import timedelta
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision,
    ShardingStrategy, StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .detection import BASE, TOKENS, ANNOTATIONS, DetectionExamples, DetectionLanguageModel
from .qualify_detection import write_json
from .tokenizer import ROOT, digest


def epoch_indices(size, world, rank, epoch, seed=42):
    if size % world or not 0 <= rank < world:
        raise ValueError('Require exact, unpadded distributed epoch coverage')
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return order[rank::world]


def learning_rate(step, total, warmup, peak):
    if not 0 <= step < total or not 0 < warmup < total:
        raise ValueError('Invalid schedule position')
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def checkpoint_context(model):
    return FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
        ShardedStateDictConfig(offload_to_cpu=True), ShardedOptimStateDictConfig(offload_to_cpu=True))


def save_checkpoint(model, opt, output, cursor, fingerprint, rank, world):
    directory = output / f"step_{cursor['updates']:06d}"
    directory.mkdir(exist_ok=True)
    with checkpoint_context(model):
        state = {'model': model.state_dict(), 'optimizer': FSDP.optim_state_dict(model, opt),
                 'rng': rng_state(), 'cursor': cursor, 'protocol_sha256': fingerprint,
                 'world_size': world, 'rank': rank}
        temporary = directory / f'rank{rank}.partial'
        torch.save(state, temporary)
        path = directory / f'rank{rank}.pth'
        temporary.replace(path)
        del state
    record = {'rank': rank, 'file': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size}
    shards = [None] * world
    dist.all_gather_object(shards, record)
    if rank == 0:
        payload = {'protocol_sha256': fingerprint, 'cursor': cursor, 'world_size': world, 'shards': shards}
        write_json(directory / 'complete.json', payload)
        # Advance latest only after every shard is durably written and hashed.
        write_json(output / 'latest.json', {'directory': directory.name, **payload})
        # Only rotate complete checkpoints produced in this new run, never other runs.
        complete = sorted(p.parent for p in output.glob('step_*/complete.json'))
        for old in complete[:-2]:
            manifest = json.loads((old / 'complete.json').read_text())
            if manifest['protocol_sha256'] != fingerprint:
                raise ValueError('Refuse to rotate a checkpoint belonging to another protocol')
            shutil.rmtree(old)
    dist.barrier()


def load_checkpoint(model, opt, output, fingerprint, rank, world):
    pointer = json.loads((output / 'latest.json').read_text())
    if pointer['protocol_sha256'] != fingerprint or pointer['world_size'] != world:
        raise ValueError('Resume protocol/world mismatch')
    directory = output / pointer['directory']
    complete = json.loads((directory / 'complete.json').read_text())
    if {k: pointer[k] for k in complete} != complete:
        raise ValueError('Incomplete checkpoint transaction')
    shard = pointer['shards'][rank]
    path = directory / shard['file']
    if shard['rank'] != rank or path.stat().st_size != shard['bytes'] or digest(path) != shard['sha256']:
        raise ValueError('Checkpoint shard integrity failure')
    with checkpoint_context(model):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['protocol_sha256'] != fingerprint or state['rank'] != rank
                or state['world_size'] != world or state['cursor'] != pointer['cursor']):
            raise ValueError('Checkpoint contents do not match committed manifest')
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(FSDP.optim_state_dict_to_load(model, opt, state['optimizer']))
        restore_rng(state['rng'])
        cursor = state['cursor']
        del state
    return cursor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rank, world, local = (int(os.environ[n]) for n in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8:
        raise ValueError('Fixed eight-rank adaptation protocol')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    device = torch.device('cuda', local)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    output = args.output.resolve()
    epochs, accumulate, peak_lr = 10, 4, 5e-5
    per_epoch = math.ceil((5400 // world) / accumulate)
    total_steps, warmup = epochs * per_epoch, per_epoch
    sources = [Path(__file__), Path(__file__).with_name('detection.py'),
               Path(__file__).with_name('attention.py'), Path(__file__).with_name('tokenizer.py'),
               Path(__file__).with_name('qualify_detection.py'), ANNOTATIONS,
               TOKENS / 'protocol.json', TOKENS / 'complete_manifest.json',
               ROOT / 'vcoco_eval/scripts/train_unihoi_detection_existing_2node8gpu.sh',
               ROOT / 'assets/unihoi_reproduction/llama3_asset_audit.json',
               ROOT / 'vcoco_eval/reports/unihoi_train_serialization_audit_20260919.json']
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        protocol = {'evidence_type': 'INDEPENDENT_UNIHOI_SUPERVISED_DETECTION_ADAPTATION',
            'full_paper_reproduction': False, 'seed': 42, 'epochs': epochs, 'world_size': world,
            'microbatch_per_gpu': 1, 'accumulation': accumulate, 'effective_batch': 32,
            'last_update_each_epoch_batch': 24, 'images_per_epoch': 5400,
            'optimizer': 'Adam', 'betas': [0.9, 0.999], 'weight_decay': 0.0,
            'peak_lr': peak_lr, 'warmup_updates': warmup, 'total_updates': total_steps,
            'lr_schedule': 'linear warmup then cosine to 0.1 * peak', 'clip_grad_norm': 1.0,
            'precision': 'FP32 master and reductions, BF16 compute',
            'fsdp': 'FULL_SHARD, gradient checkpointing, synchronize every microbatch',
            'loss': 'answer-only CE normalized by global target tokens per optimizer update',
            'checkpoint_selection': 'fixed final epoch 10, no test selection',
            'checkpoint_retention': 'latest two committed checkpoints; every 100 updates and epoch end',
            'test_labels_read': False, 'test_ap': None,
            'training_data': 'official V-COCO trainval only; no LAION/HICO mixtures',
            'independent_choices': ['JSON event serialization', 'prefix-only IAA',
                'supervised detection only, no generation/cycle/alignment/diversity losses',
                '10-epoch small-data adaptation, NOT the paper 700k-step/global512 recipe',
                '5e-5 small-data LR, NOT paper pretraining 5e-4'],
            'source_sha256': {str(f): digest(f) for f in sources},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device)}
        path = output / 'protocol.json'
        if path.exists():
            if not args.resume or json.loads(path.read_text()) != protocol:
                raise ValueError('Existing run requires explicit resume with identical protocol')
        elif args.resume:
            raise FileNotFoundError('Resume requires existing protocol and committed checkpoint')
        else:
            write_json(path, protocol)
            shutil.copytree(Path(__file__).parent, output / 'source',
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    dist.barrier()
    fingerprint = digest(output / 'protocol.json')
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    data = DetectionExamples(tokenizer)
    print(json.dumps({'rank': rank, 'stage': 'loading_pretrained_llama', 'resume': args.resume}), flush=True)
    llm, loading = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='sdpa',
        output_loading_info=True)
    for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(field):
            raise ValueError(f'Non-strict base loading: {field}: {loading[field]}')
    llm.resize_token_embeddings(len(tokenizer) + 8192)
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = FSDP(DetectionLanguageModel(llm, len(tokenizer)), device_id=device, use_orig_params=True,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD, limit_all_gathers=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.bfloat16))
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate(0, total_steps, warmup, peak_lr), foreach=False)
    cursor = {'epoch': 0, 'offset': 0, 'updates': 0}
    if args.resume:
        cursor = load_checkpoint(model, opt, output, fingerprint, rank, world)
        print(json.dumps({'rank': rank, 'stage': 'resumed', 'cursor': cursor}), flush=True)
    model.train()
    for epoch in range(cursor['epoch'], epochs):
        order = epoch_indices(len(data), world, rank, epoch)
        start_offset = cursor['offset'] if epoch == cursor['epoch'] else 0
        for offset in range(start_offset, len(order), accumulate):
            begin = time.monotonic()
            examples = [data[i] for i in order[offset:offset + accumulate]]
            tokens = torch.tensor(sum(int((x['labels'][:, 1:] != -100).sum()) for x in examples),
                                  device=device, dtype=torch.float64)
            dist.all_reduce(tokens)
            lr = learning_rate(cursor['updates'], total_steps, warmup, peak_lr)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            nll = torch.zeros((), device=device, dtype=torch.float64)
            for example in examples:
                batch = {k: v.to(device) for k, v in example.items() if torch.is_tensor(v)}
                values = model(**batch)
                loss = values['loss_sum'] * (world / tokens)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite supervised loss')
                loss.backward()
                nll += values['loss_sum'].detach().double()
            # FSDP averages each microbatch gradient; the common denominator
            # above yields exactly the global token-normalized accumulation.
            norm = model.clip_grad_norm_(1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite full-model gradient')
            opt.step()
            opt.zero_grad(set_to_none=True)
            dist.all_reduce(nll)
            cursor = {'epoch': epoch, 'offset': offset + len(examples), 'updates': cursor['updates'] + 1}
            epoch_done = cursor['offset'] == len(order)
            if epoch_done:
                cursor['epoch'], cursor['offset'] = epoch + 1, 0
            peak = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                record = {'epoch': epoch + 1, **cursor, 'display_epoch': epoch + 1,
                    'loss_per_target_token': float(nll / tokens), 'target_tokens': int(tokens),
                    'gradient_norm': float(norm), 'lr': lr, 'seconds': time.monotonic() - begin,
                    'peak_mib_max_rank': float(peak), 'test_labels_read': False}
                with (output / 'train.jsonl').open('a') as f:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
                    f.flush()
                write_json(output / 'progress.json', record)
                print(json.dumps(record), flush=True)
            if cursor['updates'] == 1 or cursor['updates'] % 100 == 0 or epoch_done:
                save_checkpoint(model, opt, output, cursor, fingerprint, rank, world)
    if rank == 0:
        write_json(output / 'training_complete.json', {'cursor': cursor, 'protocol_sha256': fingerprint,
            'final_checkpoint': json.loads((output / 'latest.json').read_text())['directory'],
            'full_unihoi_reproduction': False, 'native_test_ap': None})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
