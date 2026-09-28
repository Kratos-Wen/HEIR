"""Real Llama-3-8B/FSDP training qualification; never an AP experiment."""

import argparse
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import random
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
from .tokenizer import ROOT, digest


def write_json(path, record):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rank, world, local = (int(os.environ[n]) for n in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8:
        raise ValueError('This qualification protocol is fixed to eight ranks')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    device = torch.device('cuda', local)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'protocol.json').exists():
            raise FileExistsError('Never overwrite an earlier qualification run')
        sources = [Path(__file__), Path(__file__).with_name('detection.py'),
                   Path(__file__).with_name('attention.py'), ANNOTATIONS,
                   TOKENS / 'protocol.json', ROOT / 'assets/unihoi_reproduction/llama3_asset_audit.json']
        write_json(output / 'protocol.json', {
            'stage': 'SUPERVISED_DETECTION_REAL_MODEL_QUALIFICATION_NOT_FULL_UNIHOI',
            'seed': 42, 'world_size': world, 'batch_per_gpu': 1, 'optimizer_steps': 2,
            'checkpoint': str(BASE), 'source_sha256': {str(f): digest(f) for f in sources},
            'test_labels_read': False, 'precision': 'FP32 master, BF16 compute, FP32 reduction',
            'optimizer': 'Adam, lr=5e-4/10000 (first paper warmup step), default betas',
            'independent_choices': ['input-prefix IAA', 'JSON role serialization',
                                    'supervised CE only; no cycle/alignment/diversity claims'],
            'no_formal_training_or_test_ap': True})
    dist.barrier()
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    data = DetectionExamples(tokenizer)
    selections = data.qualification_indices()
    keys = list(selections)
    examples = [data[selections[keys[(rank + step) % len(keys)]]] for step in range(2)]
    print(json.dumps({'rank': rank, 'stage': 'loading_full_pretrained_model',
                      'cases': [e['image_id'] for e in examples]}), flush=True)
    llm, loading = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='sdpa',
        output_loading_info=True)
    for field in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if loading.get(field):
            raise ValueError(f'Non-strict pretrained loading: {field}: {loading[field]}')
    llm.resize_token_embeddings(len(tokenizer) + 8192)
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    core = DetectionLanguageModel(llm, len(tokenizer))
    model = FSDP(core, device_id=device, use_orig_params=True,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}),
        sharding_strategy=ShardingStrategy.FULL_SHARD, limit_all_gathers=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.bfloat16))
    opt = torch.optim.Adam(model.parameters(), lr=5e-4 / 10000, foreach=False)
    records = []
    for step, example in enumerate(examples):
        batch = {k: v.to(device) for k, v in example.items() if torch.is_tensor(v)}
        tokens = (batch['labels'][:, 1:] != -100).sum().float()
        dist.all_reduce(tokens)
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        values = model(**batch)
        # FSDP averages rank gradients; normalize by global supervised tokens.
        loss = values['loss_sum'] * world / tokens
        loss.backward()
        norm = model.clip_grad_norm_(1.0)
        if not torch.isfinite(norm):
            raise ValueError('Non-finite full-model gradient')
        grads = torch.zeros(3, device=device, dtype=torch.float64)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise ValueError('Non-finite parameter gradient: ' + name)
                group = 0 if 'prefix_adapter' in name else 1 if 'modality' in name else 2
                grads[group] += parameter.grad.detach().abs().sum().double()
        dist.all_reduce(grads)
        if not (grads > 0).all():
            raise ValueError('Missing gradient in IAA, modality embeddings, or Llama')
        opt.step()
        record = {'rank': rank, 'step': step, 'image_id': example['image_id'],
                  'loss': float(loss), 'gradient_norm': float(norm),
                  'branch_gradient_l1': grads.tolist(), 'seconds': time.monotonic() - start,
                  'sequence_length': batch['input_ids'].shape[1],
                  'global_supervised_tokens': int(tokens),
                  'peak_mib': torch.cuda.max_memory_allocated() / 1024**2}
        records.append(record)
        write_json(output / f'rank{rank}_progress.json', records)
        print(json.dumps(record), flush=True)
    # Same-process distributed save/reload check, not a crash-restart claim.
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        reference = model(**batch)['loss_sum'].detach().clone()
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True), ShardedOptimStateDictConfig(offload_to_cpu=True)):
        state = {'model': model.state_dict(), 'optimizer': FSDP.optim_state_dict(model, opt),
                 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'steps': 2}
        tmp = output / f'rank{rank}_checkpoint.partial'
        torch.save(state, tmp)
        checkpoint_path = output / f'rank{rank}_checkpoint.pth'
        tmp.replace(checkpoint_path)
        del state
        dist.barrier()
        restored = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(restored['model'], strict=True)
        opt.load_state_dict(FSDP.optim_state_dict_to_load(model, opt, restored['optimizer']))
        torch.set_rng_state(restored['rng'])
        torch.cuda.set_rng_state(restored['cuda_rng'])
        del restored
    with torch.no_grad():
        reloaded = model(**batch)['loss_sum']
    torch.testing.assert_close(reloaded, reference, rtol=0, atol=0)
    result = {'rank': rank, 'records': records, 'checkpoint_save_reload_exact': True,
              'checkpoint_sha256': digest(checkpoint_path)}
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        write_json(output / 'qualification.json', {'status': 'PASS', 'world_size': world,
            'real_pretrained_llama': True, 'strict_base_weights': True, 'cases': selections,
            'ranks': gathered, 'full_unihoi_training_complete': False, 'native_test_ap': None})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
