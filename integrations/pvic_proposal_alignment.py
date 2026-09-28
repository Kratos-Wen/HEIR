"""Preserve H-DETR query identity through its native top-k postprocessor.

The pinned H-DETR postprocessor gathers boxes by a flattened query/class
top-k. PViC's proposal utility indexes the supplied features by *detection
row*. Therefore features must undergo the same gather before NMS. This module
does not change a box, class, score, threshold, NMS rule, or learned weight.
"""

from __future__ import annotations

from functools import partial

import torch


PROPOSAL_FEATURE_ALIGNMENT_MODES = ("legacy_row", "query_index")


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
    gather = indices[None, :, :, None].expand(
        states.shape[0], -1, -1, states.shape[-1]
    )
    return torch.gather(states, 2, gather)


def _aligned_forward(detector, images, *, native_forward, topk):
    outputs, states, features = native_forward(detector, images)
    aligned = align_detector_query_states(outputs["pred_logits"], states, topk)
    return outputs, aligned, features


def configure_pvic_proposal_alignment(model, mode):
    if mode not in PROPOSAL_FEATURE_ALIGNMENT_MODES:
        raise ValueError("Unknown proposal feature alignment: " + mode)
    current = getattr(model, "proposal_feature_alignment", "legacy_row")
    if current == mode:
        return model
    if current != "legacy_row" or mode != "query_index":
        raise ValueError("Do not switch proposal semantics on an initialized model.")
    if not hasattr(model.postprocessor, "topk"):
        raise ValueError("Query alignment requires H-DETR's top-k postprocessor.")
    model.od_forward = partial(
        _aligned_forward,
        native_forward=model.od_forward,
        topk=int(model.postprocessor.topk),
    )
    model.proposal_feature_alignment = mode
    return model


def validate_checkpoint_proposal_alignment(checkpoint, expected_mode):
    recorded = checkpoint.get("proposal_feature_alignment", "legacy_row")
    if recorded != expected_mode:
        raise ValueError(
            "Checkpoint proposal features use {} but runtime requests {}. "
            "Retrain the head with corrected features; do not silently change "
            "an old checkpoint's inference semantics.".format(recorded, expected_mode)
        )


__all__ = [
    "PROPOSAL_FEATURE_ALIGNMENT_MODES",
    "align_detector_query_states",
    "configure_pvic_proposal_alignment",
    "detector_topk_query_indices",
    "validate_checkpoint_proposal_alignment",
]
