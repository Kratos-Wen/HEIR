"""Pinned detector identity fix; unchanged CoRISP proposal scores/geometry/NMS."""
import hashlib
import importlib.util

import torch

from .environment import ROOT
from hdetr_corisp_v3 import _hdetr_forward_with_tokens, _normalize_proposal_precision
from ops import prepare_region_proposals

ALIGNMENT_SOURCE = ROOT/'integrations/pvic_proposal_alignment.py'
ALIGNMENT_SHA256 = 'd8087e7dfb219d8461233e7dcfe886236b6af1fc6426a66afa6a71e5fee2c687'
if hashlib.sha256(ALIGNMENT_SOURCE.read_bytes()).hexdigest() != ALIGNMENT_SHA256:
    raise ValueError('Audited detector identity helper changed')
spec = importlib.util.spec_from_file_location('_heir_pinned_proposal_alignment', ALIGNMENT_SOURCE)
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def prepare(raw, hidden, postprocessor, sizes, settings):
    aligned = alignment.align_detector_query_states(raw['pred_logits'], hidden, postprocessor.topk)
    detections = postprocessor(raw, sizes)
    proposals = prepare_region_proposals(detections, aligned[-1], sizes, **settings)
    return _normalize_proposal_precision(proposals)


def extract(model, images):
    sizes = torch.as_tensor([image.shape[-2:] for image in images], device=images[0].device)
    with torch.no_grad():
        raw, hidden, features = _hdetr_forward_with_tokens(model.detector, images)
        proposals = prepare(raw, hidden, model.postprocessor, sizes, dict(box_score_thresh=model.box_score_thresh,
            human_idx=model.human_idx, min_instances=model.min_instances, max_instances=model.max_instances))
    packets = model.adapter(proposals, features[-1].tensors, sizes,
        dense_valid_masks=~features[-1].mask, source='frozen_h_detr_swin_l_corisp_only')
    pairs = [torch.stack([p.subject_indices[0], p.object_indices[0]], -1) for p in packets]
    return sizes, proposals, [p['boxes'] for p in proposals], pairs, [], [], packets
