"""Isolated GroupHOI-S runtime; upstream checkout remains unchanged.

Missing author dense-CLIP interface is reconstructed explicitly. The matcher
bridge fixes a QBC/BQC contract error, not the matching objective.
"""

import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn

from .clip_features import load_verified_b16
from .postprocess import AUTHOR, ROOT

DATA = ROOT.parent / 'data/v-coco'
ASSETS = ROOT / 'assets/group_hoi_reproduction'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def configuration():
    tree = ast.parse((AUTHOR / 'main.py').read_text())
    node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_args_parser']
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(AUTHOR / 'main.py'), 'exec'), scope)
    args = scope['get_args_parser']().parse_args([
        '--hoi', '--dataset_file', 'vcoco', '--hoi_path', str(DATA),
        '--num_obj_classes', '81', '--num_verb_classes', '29', '--batch_size', '4',
        '--use_nms_filter', '--ft_clip_with_small_lr', '--with_clip_label',
        '--with_obj_clip_label', '--n_layer', '3'])
    return args


class BatchMajorMatcher(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, outputs, targets):
        adapted = dict(outputs)
        logits = outputs['pred_hoi_logits']
        b, q = outputs['pred_sub_boxes'].shape[:2]
        if logits.shape[:2] != (q, b):
            raise ValueError('Expected native QBC classification logits')
        adapted['pred_hoi_logits'] = logits.transpose(0, 1)
        return self.native(adapted, targets)


def build_runtime(device):
    if any(name in sys.modules for name in ('models', 'datasets', 'util')):
        raise RuntimeError('Use a clean dedicated GroupHOI process')
    sys.path.insert(0, str(AUTHOR))
    package = types.ModuleType('models')
    package.__path__ = [str(AUTHOR / 'models')]
    sys.modules['models'] = package
    vlm, preprocess = load_verified_b16(ASSETS / 'ViT-B-16.pt', device)

    def load_model_and_preprocess(name, model_type, is_eval, device):
        if (name, model_type, is_eval) != ('clip_feature_extractor', 'ViT-B-16', True):
            raise ValueError('Unexpected author CLIP request')
        return vlm, {'eval': preprocess}, {'eval': str}

    source = AUTHOR / 'models/grouphoi.py'
    tree = ast.parse(source.read_text())
    removed = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'lavis.models']
    if len(removed) != 1:
        raise ValueError('Upstream CLIP import changed')
    tree.body.remove(removed[0])
    module = types.ModuleType('models.grouphoi')
    module.__package__ = 'models'
    module.__file__ = str(source)
    module.load_model_and_preprocess = load_model_and_preprocess
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    args = configuration()
    args.device = str(device)
    model, criterion, _, _, _, vis, txt = module.build(args)
    model = model.to(device)
    criterion.matcher = BatchMajorMatcher(criterion.matcher)
    return model, criterion, vlm, args, vis, txt


def load_detr(model):
    source = ASSETS / 'detr-r50-e632da11.pth'
    if not sha(source).startswith('e632da11'):
        raise ValueError('Official DETR filename hash mismatch')
    weights = torch.load(source, map_location='cpu', weights_only=False)['model']
    mapped = {}
    ignored = []
    state = model.state_dict()
    for name, value in weights.items():
        names = [name]
        if name.startswith('transformer.decoder.'):
            names = [name.replace('transformer.decoder.', 'transformer.instance_decoder.')]
        elif name.startswith('bbox_embed.'):
            names = [name.replace('bbox_embed.', p) for p in ('hum_bbox_embed.', 'obj_bbox_embed.')]
        for target in names:
            if target in state:
                if state[target].shape != value.shape:
                    raise ValueError(f'DETR tensor shape mismatch: {target}')
                mapped[target] = value
            elif not (name.startswith(('query_embed.', 'class_embed.')) or
                      name.startswith(tuple(f'transformer.decoder.layers.{i}.' for i in (3, 4, 5)))):
                raise ValueError(f'Unclassified DETR unused tensor: {name} -> {target}')
            else:
                ignored.append(name)
    required = [k for k in state if k.startswith(('backbone.', 'input_proj.',
                'transformer.encoder.', 'transformer.instance_decoder.',
                'hum_bbox_embed.', 'obj_bbox_embed.')) and
                not any(part in k for part in ('.graph_layer.', '.graph_layers.', '.norm_graph.'))]
    missing = sorted(set(required) - set(mapped))
    if missing:
        raise ValueError(f'Incomplete DETR initialization: {missing}')
    result = model.load_state_dict(mapped, strict=False)
    if result.unexpected_keys:
        raise ValueError(result.unexpected_keys)
    return {'asset_sha256': sha(source), 'loaded': sorted(mapped),
            'unused_author_converter_keys': sorted(set(ignored)),
            'new_parameters': result.missing_keys,
            'note': 'Original converter does not map query_embed to h/o queries or initialize 768d interaction decoder.'}


def optimizer(model, args):
    groups = [[], [], []]
    names = [[], [], []]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        i = 1 if 'backbone' in name else (2 if 'visual_projection' in name else 0)
        groups[i].append(parameter)
        names[i].append(name)
    result = torch.optim.AdamW([
        {'params': group, 'lr': lr} for group, lr in zip(groups, (args.lr, args.lr_backbone, args.lr_clip))
    ], weight_decay=args.weight_decay)
    return result, names


def train_dataset(vis, txt, args):
    from datasets.vcoco import build
    data = build(vis, txt, 'train', args)
    expected = {int(x) for x in (DATA / 'data/splits/vcoco_trainval.ids').read_text().split()}
    actual = [int(Path(x['file_name']).stem.rsplit('_', 1)[-1]) for x in data.annotations]
    if len(actual) != len(set(actual)) or set(actual) != expected or len(expected) != 5400:
        raise ValueError('Trainval ID mismatch')
    return data


def batch_to_device(batch, device):
    samples, targets = batch[:2]
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    return samples.to(device), targets, torch.stack([t['clip_inputs'] for t in targets])


def total_loss(criterion, outputs, targets):
    losses = criterion(outputs, targets)
    value = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite GroupHOI loss')
    return value, losses
