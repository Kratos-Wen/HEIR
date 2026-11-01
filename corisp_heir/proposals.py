"""Align detector query states with proposals; scores, geometry and NMS are unchanged.

The pinned H-DETR postprocessor gathers boxes by a flattened query/class top-k. PViC's proposal utility
indexes the supplied features by detection row, so features must undergo the same gather before NMS.
"""
import torch


def detector_topk_query_indices(logits, topk):
    if logits.ndim != 3 or topk <= 0:
        raise ValueError("Detector logits must be [B,Q,C] and topk positive.")
    if not torch.isfinite(logits).all():
        raise ValueError("Detector logits contain nonfinite values.")
    # Use exactly the pinned H-DETR PostProcess operation, including ties and
    # repeated query IDs when several noun hypotheses survive for one query.
    indices = torch.topk(logits.sigmoid().flatten(1), topk, dim=1).indices
    return indices // logits.shape[-1]


def align_detector_query_states(logits, states, topk):
    if states.ndim != 4 or states.shape[1] != logits.shape[0]:
        raise ValueError("Decoder states must be [layers,B,Q,D].")
    if states.shape[2] < logits.shape[1]:
        raise ValueError("Decoder states do not cover the one-to-one queries.")
    indices = detector_topk_query_indices(logits, topk)
    gather = indices[None, :, :, None].expand(states.shape[0], -1, -1, states.shape[-1])
    return torch.gather(states, 2, gather)

