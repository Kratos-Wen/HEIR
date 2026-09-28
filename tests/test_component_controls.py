import copy

import torch

from tests.test_corisp_agent_role_field import _model, _packet, _inputs
from corisp import RoleArityEventField, RoleArityEventFieldConfig
from corisp_heir.batched_execution import potentials


def test_no_role_feedback_removes_role_influence_only_from_recurrence():
    base = _model(enable_typed_null_fillers=False).eval()
    base.residual_scale.data.fill_(.5)
    changed = copy.deepcopy(base)
    changed.role_residual.weight.data.normal_(std=1.)
    visual, semantic, aligned = _inputs()
    def run(model):
        return model(_packet(), visual_grid=visual, global_semantic=semantic, aligned_agent_tokens=aligned)
    assert not torch.allclose(run(base)['event_states'], run(changed)['event_states'])
    for model in (base, changed):
        model.disable_role_feedback = True
    first, second = run(base), run(changed)
    torch.testing.assert_close(first['event_states'], second['event_states'], atol=0, rtol=0)
    assert not torch.allclose(first['joint_role_probs'], second['joint_role_probs'])
    assert set(base.state_dict()) == set(changed.state_dict())


def test_heir_potentials_remain_fp32_under_autocast():
    field = RoleArityEventField(RoleArityEventFieldConfig(d_model=8, num_roles=2,
        max_cardinality=5, num_heads=2, arity_rank=4, dropout=0.)).eval()
    field.arity_log_scale.data.fill_(.3)
    field.cardinality_head.weight.data.normal_(std=.05)
    visible = torch.randn(2, 3, 3).softmax(-1)[..., 1:]
    states, context = torch.randn(2, 8), torch.randn(2, 2, 8)
    expected = potentials(field, visible, states, context)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = potentials(field, visible, states, context)
    for before, after in zip(expected, actual):
        assert after.dtype == torch.float32
        torch.testing.assert_close(before, after, atol=0, rtol=0)
