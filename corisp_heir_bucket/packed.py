"""Image-wide exact event buckets; no cross-person probability coupling.

Batch only events of equal candidate/role count. Identity and loss order remain
person-action keyed. Structured potentials use FP32; visual and contextual
features use BF16 autocast.
"""
import torch
from . import execution, vectorized


def terms(model, visible, states, context, support, descriptions):
    field, role_sets = model.role_arity_event_field, model.heir_active_roles
    ordered = {}
    counts = dict(events=0, recurrence_batches=0, role_count_states=0)
    with torch.autocast(visible.device.type, enabled=False):
        for desc in execution.chunks(descriptions, role_sets, getattr(model, 'heir_packed_chunk', 64)):
            vv = torch.stack([visible[d[2], d[1]] for d in desc])
            ee = torch.stack([states[d[0], d[1]] for d in desc])
            cc = torch.stack([context[d[0], d[1]] for d in desc])
            ss = torch.stack([support[d[2], d[1]] for d in desc])
            loss = vectorized.losses(field, vv, ee, cc, desc, role_sets, ss, model.alpha, model.gamma)
            ordered.update({(d[0], d[1]): value for d, value in zip(desc, loss.unbind())})
            counts['events'] += len(desc)
            counts['recurrence_batches'] += 1
            counts['role_count_states'] += len(desc)*3**len(role_sets[desc[0][1]])
    return ordered, counts


def forward(model, images, targets=None):
    return execution.forward(model, images, targets, term_runner=terms, whole_image=True)
