"""Support conditioning: energy, likelihood, gradient and role-ID parity."""
import copy
import io
import os
from unittest.mock import patch

import pytest
import torch

from corisp_heir import environment
from corisp_heir.compatibility import Compatibility, digest, encoded
from corisp_heir.exact_dp import partition
from corisp_heir.inference import potentials
from corisp_heir import packed
from corisp_heir.support import aligned_support, state_indices, target_mass
from corisp import EventSetField, EventSetFieldConfig, TargetGroup
from tests.test_exact_dp import reference


def field_with_potentials(seed):
    torch.manual_seed(seed)
    field = EventSetField(EventSetFieldConfig(d_model=8, num_roles=6, max_cardinality=8, num_heads=2,
                                              arity_rank=4, dropout=0.)).eval()
    with torch.no_grad():
        field.arity_log_scale.fill_(.4)
        field.cardinality_head.weight.normal_(std=.05)
    return field


def likelihood(field, visible, states, contexts, descriptions, active, support, solve=partition):
    weights, energy = potentials(field, visible, states, contexts, active, support)
    z, coefficients = solve(weights, energy)
    terms = []
    for j, (human, action, indices, groups) in enumerate(descriptions):
        mass = target_mass(field, human, action, indices, groups, active, weights[j], energy[j], coefficients[j], z[j])
        terms.append(.5*field.focal_transform_nll((z[j]-mass).clamp_min(0), .1))
    return torch.stack(terms)


def test_execution_setting_follows_the_checkpoint():
    with patch.dict(os.environ, HEIR_CORISP_COMPILE_DP='1'):
        environment.restore_execution({})
        assert os.environ['HEIR_CORISP_COMPILE_DP'] == '0'
        environment.restore_execution(dict(compiled_dp=True))
        assert os.environ['HEIR_CORISP_COMPILE_DP'] == '1'


def test_release_person_first_mapping():
    document = dict(status='HUMAN_APPROVED', null_state_allowed=True,
                    vocabulary=dict(verbs=['hold', 'sit'], nouns=['chair', 'person'], roles=['target', 'support']),
                    vocabulary_sha256='synthetic', role_bits_verb_noun=[[1, 1], [2, 0]])
    document['content_sha256'] = digest(encoded(document))
    table = Compatibility(document)
    axes = dict(table.axes, nouns=['person', 'chair'])
    mask, active = aligned_support(table, axes)
    assert axes['nouns'][0] == 'person'
    assert int(mask.sum()) == 3
    for ni, noun in enumerate(axes['nouns']):
        for vi, verb in enumerate(axes['verbs']):
            assert mask[ni, vi].tolist() == [table.allows(verb, noun, r) for r in axes['roles']]
    assert all(active)
    with pytest.raises(ValueError):
        aligned_support(table, dict(axes, nouns=axes['nouns'][:-1]))


def test_changed_compatibility_content_is_rejected():
    document = dict(status='HUMAN_APPROVED', null_state_allowed=True,
                    vocabulary=dict(verbs=['hold'], nouns=['person'], roles=['target']),
                    vocabulary_sha256='synthetic', role_bits_verb_noun=[[1]])
    document['content_sha256'] = digest(encoded(document))
    document['role_bits_verb_noun'] = [[0]]
    with pytest.raises(ValueError, match='Content hash mismatch'):
        Compatibility(document)


@pytest.mark.parametrize('active', [(5,), (1, 4), (0, 2, 5), tuple(range(6))])
def test_dense_compact_loss_gradient(active):
    field = field_with_potentials(433)
    other = copy.deepcopy(field)
    raw = torch.randn(2, 4, 7, requires_grad=True)
    clone = raw.detach().clone().requires_grad_()
    e, c = torch.randn(2, 8), torch.randn(2, 6, 8)
    mask = torch.zeros(2, 4, 6, dtype=torch.bool)
    mask[:, :, list(active)] = True
    mask[:, 3, :] = False
    groups = [TargetGroup(active[-1], torch.tensor([0, 1])), TargetGroup(active[-1], torch.tensor([0, 1]))]
    desc = [(0, 0, torch.arange(4), groups), (0, 1, torch.arange(4), [])]
    dense = likelihood(field, raw.softmax(-1)[..., 1:], e, c, desc, tuple(range(6)), mask)
    compact = likelihood(other, clone.softmax(-1)[..., 1:], e, c, desc, active, mask)
    torch.testing.assert_close(dense, compact, atol=3e-6, rtol=3e-6)
    dense.sum().backward()
    compact.sum().backward()
    torch.testing.assert_close(raw.grad, clone.grad, atol=4e-6, rtol=4e-5)
    for a, b in zip(field.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.isfinite(b.grad).all()
            torch.testing.assert_close(a.grad, b.grad, atol=4e-6, rtol=4e-5)


def test_training_batches_match_the_per_event_likelihood():
    field = field_with_potentials(433)
    other = copy.deepcopy(field)
    active = (1, 4)
    role_sets = [active, active]
    raw = torch.randn(2, 4, 7, requires_grad=True)
    clone = raw.detach().clone().requires_grad_()
    e, c = torch.randn(2, 8), torch.randn(2, 6, 8)
    mask = torch.zeros(2, 4, 6, dtype=torch.bool)
    mask[:, :, list(active)] = True
    groups = [TargetGroup(1, torch.tensor([0, 1])), TargetGroup(4, torch.tensor([2]))]
    desc = [(0, 0, torch.arange(4), groups), (0, 1, torch.arange(4), [])]
    event = likelihood(field, raw.softmax(-1)[..., 1:], e, c, desc, active, mask)
    batched = packed.losses(other, clone.softmax(-1)[..., 1:], e, c, desc, role_sets, mask, .5, .1)
    torch.testing.assert_close(event, batched, atol=3e-6, rtol=3e-6)
    event.sum().backward()
    batched.sum().backward()
    torch.testing.assert_close(raw.grad, clone.grad, atol=4e-6, rtol=4e-5)


def test_set_likelihood_updates_and_restore():
    field = field_with_potentials(426)
    other = copy.deepcopy(field)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    clone = logits.detach().clone().requires_grad_()
    params = [list(field.parameters())+[logits], list(other.parameters())+[clone]]
    optimizers = [torch.optim.AdamW(p, lr=1e-4) for p in params]
    states, contexts = torch.randn(2, 8), torch.randn(2, 6, 8)
    support = torch.ones(2, 4, 6, dtype=torch.bool)
    active = tuple(range(6))
    groups = [TargetGroup(0, torch.tensor([0, 1])), TargetGroup(0, torch.tensor([0, 1])),
              TargetGroup(4, torch.tensor([2, 3]))]
    descriptions = [(0, 0, torch.arange(4), groups), (0, 1, torch.arange(4), [])]
    for step in range(3):
        values = []
        for solve, model, raw, optimizer in zip((reference, partition), (field, other), (logits, clone), optimizers):
            optimizer.zero_grad(set_to_none=True)
            value = likelihood(model, raw.softmax(-1)[..., 1:], states, contexts, descriptions, active, support, solve).sum()
            value.backward()
            values.append(value.detach())
        torch.testing.assert_close(*values, atol=3e-6, rtol=3e-6)
        for a, b in zip(*params):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)
        for optimizer, parameters in zip(optimizers, params):
            torch.nn.utils.clip_grad_norm_(parameters, .1, error_if_nonfinite=True)
            optimizer.step()
        for a, b in zip(*params):
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        if step == 1:
            buffer = io.BytesIO()
            torch.save(dict(model=other.state_dict(), optimizer=optimizers[1].state_dict()), buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            other.load_state_dict(saved['model'], strict=True)
            optimizers[1].load_state_dict(saved['optimizer'])


def test_energy_is_full_submatrix_and_marginals():
    torch.manual_seed(13)
    f = EventSetField(EventSetFieldConfig(d_model=8, num_roles=6, max_cardinality=8, num_heads=2,
                                          arity_rank=4, dropout=0.)).eval()
    with torch.no_grad():
        f.arity_log_scale.fill_(.5)
    active = (1, 3, 5)
    mask = torch.zeros(1, 3, 6, dtype=torch.bool)
    mask[:, :, list(active)] = True
    v = torch.randn(1, 3, 7).softmax(-1)[..., 1:]
    e, c = torch.randn(1, 8), torch.randn(1, 6, 8)
    wd, ed = potentials(f, v, e, c, tuple(range(6)), mask)
    wc, ec = potentials(f, v, e, c, active, mask)
    _, codes = state_indices(active, 6, e.device)
    torch.testing.assert_close(ec, ed[:, :, codes], atol=2e-6, rtol=2e-6)
    zd, _ = partition(wd.requires_grad_(), ed)
    zc, _ = partition(wc.requires_grad_(), ec)
    gd = torch.autograd.grad(zd.sum(), wd)[0]
    gc = torch.autograd.grad(zc.sum(), wc)[0]
    torch.testing.assert_close(gd[:, :, [0, 2, 4, 6]], gc, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(gc.sum(-1), torch.ones(1, 3), atol=2e-6, rtol=2e-6)
    assert float(gd[:, :, [1, 3, 5]].abs().sum()) == 0.
