"""Frozen official DDETR-R50 asset, not a claim about the paper's detector.

Legacy import compatibility is isolated here. Frozen inference uses the
author-supplied PyTorch deformable-attention reference on CUDA, not a new op.
"""

import argparse
import ast
import hashlib
import importlib
from pathlib import Path
import sys
import types

import torch
from torchvision.ops import batched_nms, box_convert, nms, roi_align

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / 'reference_repos/Deformable-DETR'
WEIGHTS = ROOT / 'assets/hoi_idiff_reproduction/r50_deformable_detr_plus_iterative_bbox_refinement_official.pth'
WEIGHT_SHA = '64865b1d03d5f421eb12d243917e72c003467b630434c0ae79d256309f2dba98'
COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86,
    87, 88, 89, 90)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def install_reference_operator():
    source = AUTHOR / 'models/ops/functions/ms_deform_attn_func.py'
    tree = ast.parse(source.read_text())
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ms_deform_attn_core_pytorch']
    scope = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    core = scope[fn.name]
    extension = types.ModuleType('MultiScaleDeformableAttention')
    extension.ms_deform_attn_forward = lambda v, shapes, starts, loc, attn, chunk: core(v, shapes, loc, attn)

    def forbidden_backward(*args):
        raise RuntimeError('Reference detector backend is frozen-inference-only')

    extension.ms_deform_attn_backward = forbidden_backward
    sys.modules['MultiScaleDeformableAttention'] = extension


def build_detector(device):
    if digest(WEIGHTS) != WEIGHT_SHA:
        raise ValueError('Official DDETR asset hash mismatch')
    if any(k in sys.modules for k in ('models', 'util')):
        raise RuntimeError('Detector requires an isolated process namespace')
    sys.path.insert(0, str(AUTHOR))
    install_reference_operator()
    importlib.import_module('util')
    path = AUTHOR / 'util/misc.py'
    tree = ast.parse(path.read_text())
    # Upstream parses torchvision 0.20 as float(0.2), incorrectly selecting
    # removed pre-0.5 APIs. Modern torchvision needs neither legacy branch.
    legacy = [n for n in tree.body if isinstance(n, ast.If) and 'torchvision.__version__' in ast.unparse(n.test)]
    if len(legacy) != 1:
        raise ValueError('Unexpected legacy torchvision compatibility structure')
    tree.body.remove(legacy[0])
    module = types.ModuleType('util.misc')
    module.__file__, module.__package__ = str(path), 'util'
    sys.modules['util.misc'] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    config = state['args']
    config.device = str(device)
    backbone = importlib.import_module('models.backbone')
    backbone.is_main_process = lambda: False  # Every tensor is strictly restored below.
    build = importlib.import_module('models.deformable_detr').build
    model, _, _ = build(config)
    result = model.load_state_dict(state['model'], strict=True)
    model = model.eval().requires_grad_(False).to(device)
    return model, {'asset_sha256': WEIGHT_SHA, 'state_tensors': len(state['model']),
        'missing_keys': result.missing_keys, 'unexpected_keys': result.unexpected_keys,
        'frozen': not any(p.requires_grad for p in model.parameters()),
        'attention_backend': 'unchanged author PyTorch reference, CUDA, inference-only',
        'paper_detector_equivalence': False, 'pretraining': 'generic COCO detector, not V-COCO-specific'}


@torch.no_grad()
def extract(model, tensor):
    if tensor.ndim != 3:
        raise ValueError('One unpadded image per frozen detector forward')
    saved = {}
    hooks = [model.transformer.register_forward_hook(lambda m, i, o: saved.update(queries=o[0][-1][0])),
             model.input_proj[2].register_forward_hook(lambda m, i, o: saved.update(spatial=o))]
    try:
        prediction = model([tensor])
    finally:
        for hook in hooks:
            hook.remove()
    probabilities = prediction['pred_logits'][0, :, list(COCO_IDS)].sigmoid()
    boxes = box_convert(prediction['pred_boxes'][0], 'cxcywh', 'xyxy').clamp(0, 1)
    scores, labels = probabilities.max(-1)
    valid = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    humans = torch.where(valid & (probabilities[:, 0] >= .2))[0]
    humans = humans[nms(boxes[humans], probabilities[humans, 0], .6)][:10]
    objects = torch.where(valid & (scores >= .2))[0]
    objects = objects[batched_nms(boxes[objects], scores[objects], labels[objects], .6)][:30]
    pairs = [(int(h), int(o)) for h in humans for o in objects if h != o]
    pairs += [(int(h), -1) for h in humans]
    n = len(pairs)
    if not n:
        return {'human_boxes': torch.empty(0, 4), 'entity_boxes': torch.empty(0, 4),
            'human_query': torch.empty(0, dtype=torch.long), 'entity_query': torch.empty(0, dtype=torch.long),
            'appearance': torch.empty(0, 777), 'prior': torch.empty(0, 81),
            'detection_score': torch.empty(0), 'predicted_noun': torch.empty(0, dtype=torch.long),
            'null': torch.empty(0, dtype=torch.bool)}
    hids = torch.tensor([h for h, o in pairs], device=tensor.device)
    oids = torch.tensor([o for h, o in pairs], device=tensor.device)
    null = oids < 0
    hb, ob = boxes[hids], boxes[oids.clamp_min(0)].clone()
    ob[null] = 0
    union = torch.cat((torch.minimum(hb[:, :2], ob[:, :2]), torch.maximum(hb[:, 2:], ob[:, 2:])), -1)
    union[null] = hb[null]
    spatial = saved['spatial']
    scale = union.new_tensor([spatial.shape[-1], spatial.shape[-2]] * 2)
    region = roi_align(spatial, [union * scale], output_size=7, spatial_scale=1, aligned=True).mean((2, 3))
    ofeat = saved['queries'][oids.clamp_min(0)].clone()
    ofeat[null] = 0
    app = torch.cat((saved['queries'][hids], ofeat, region, hb, ob, null[:, None].float()), -1)
    prior = probabilities[oids.clamp_min(0)]
    prior = torch.cat((prior / prior.sum(-1, keepdim=True).clamp_min(1e-12), prior.new_zeros(n, 1)), -1)
    prior[null] = 0
    prior[null, 80] = 1
    noun = labels[oids.clamp_min(0)].clone()
    noun[null] = 80
    entity_score = scores[oids.clamp_min(0)].clone()
    entity_score[null] = 1
    result = {'human_boxes': hb, 'entity_boxes': ob, 'human_query': hids, 'entity_query': oids,
        'appearance': app, 'prior': prior, 'predicted_noun': noun, 'null': null,
        'detection_score': probabilities[hids, 0] * entity_score}
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite frozen detector feature')
    return {k: v.cpu() for k, v in result.items()}
