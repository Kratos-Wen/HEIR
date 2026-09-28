"""Reuse the released cache postprocessor, with an explicit QBC -> BQC bridge."""

import ast
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_convert


ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / 'reference_repos/GroupHOI'
EXPORT_SHA256 = '1980a80729af15248e8bffa2a384bd358f0f22b4957b1cff4bf191e8a6e2977a'


def author_cache_processor(args, correct_mat):
    source = AUTHOR / 'generate_vcoco_official.py'
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPORT_SHA256:
        raise ValueError('Upstream exporter changed; re-audit before use')
    # Loading only this unchanged class avoids constructing its obsolete model
    # and importing unrelated/missing training dependencies.
    tree = ast.parse(raw)
    cls, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PostProcessHOI']
    label_tree = ast.parse((AUTHOR / 'datasets/vcoco_text_label.py').read_text())
    labels, = [ast.literal_eval(n.value) for n in label_tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'vcoco_hoi_text_label' for t in n.targets)]
    namespace = {'torch': torch, 'nn': nn, 'np': np, 'F': F,
                 'defaultdict': defaultdict, 'vcoco_hoi_text_label': labels,
                 'box_cxcywh_to_xyxy': lambda b: box_convert(b, 'cxcywh', 'xyxy')}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['PostProcessHOI'](args.num_queries, args.subject_category_id, correct_mat, args)


class GroupCachePostprocessor(nn.Module):
    def __init__(self, args, correct_mat):
        super().__init__()
        self.author = author_cache_processor(args, correct_mat)

    @torch.no_grad()
    def forward(self, outputs, sizes):
        # Current Group_HOI emits QBC HOI logits, but BQC objects/boxes.
        adapted = dict(outputs)
        hoi = outputs['pred_hoi_logits']
        obj = outputs['pred_obj_logits']
        if hoi.ndim != 3 or obj.ndim != 3 or hoi.shape[:2] != (obj.shape[1], obj.shape[0]):
            raise ValueError('Expected native GroupHOI QBC logits and BQC object outputs')
        adapted['pred_hoi_logits'] = hoi.transpose(0, 1)
        return self.author(adapted, sizes)
