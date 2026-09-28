import itertools
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import (  # noqa: E402
    GroundedRoleSetConfig,
    GroundedRoleSetAblation,
    GroundedRoleSetField,
    RoleFillerTargetGroup,
)


VALID_ROLES = torch.tensor(
    [[True, False], [True, True], [False, True]], dtype=torch.bool
)


def _field() -> GroundedRoleSetField:
    torch.manual_seed(811)
    return GroundedRoleSetField(
        GroundedRoleSetConfig(d_model=8, num_roles=2, max_cardinality=8)
    )


def _inputs():
    visible = torch.tensor(
        [
            [[0.20, 0.00], [0.10, 0.15], [0.00, 0.30]],
            [[0.35, 0.00], [0.05, 0.20], [0.00, 0.10]],
            [[0.15, 0.00], [0.25, 0.10], [0.00, 0.25]],
        ],
        dtype=torch.float32,
    )
    typed = torch.tensor(
        [
            [[0.10, 0.00], [0.12, 0.18], [0.00, 0.14]],
            [[0.08, 0.00], [0.16, 0.09], [0.00, 0.20]],
        ],
        dtype=torch.float32,
    )
    return {
        "visible_role_probs": visible,
        "typed_null_role_probs": typed,
        "pair_agent_indices": torch.tensor([0, 0, 1]),
        "event_states": torch.randn(2, 3, 8),
        "visible_action_mask": torch.ones(3, 3, dtype=torch.bool),
        "valid_role_mask": VALID_ROLES,
    }


def _exhaustive(log_states, cardinality):
    assignments = list(
        itertools.product(range(log_states.shape[1]), repeat=log_states.shape[0])
    )
    energies = []
    for assignment in assignments:
        state = torch.tensor(assignment, device=log_states.device)
        energy = log_states[
            torch.arange(log_states.shape[0], device=log_states.device), state
        ].sum()
        energy = energy + cardinality[int((state > 0).sum())]
        energies.append(energy)
    energies = torch.stack(energies)
    log_z = torch.logsumexp(energies, dim=0)
    probability = torch.softmax(energies, dim=0)
    marginal = torch.zeros_like(log_states)
    for assignment, weight in zip(assignments, probability):
        for row, state in enumerate(assignment):
            marginal[row, state] += weight
    return log_z, marginal


def test_dynamic_program_matches_exhaustive_partition_and_marginals():
    log_states = torch.log(
        torch.tensor(
            [
                [0.55, 0.30, 0.15],
                [0.60, 0.40, 0.00],
                [0.50, 0.00, 0.50],
            ]
        )
    )
    cardinality = torch.tensor([0.0, 0.3, -0.2, -0.7])
    expected_z, expected_marginal = _exhaustive(log_states, cardinality)
    actual_z, _ = GroundedRoleSetField.log_partition(log_states, cardinality)
    assert torch.allclose(actual_z, expected_z, atol=1e-6)

    leaf = log_states.clone().requires_grad_(True)
    differentiable_z, _ = GroundedRoleSetField.log_partition(leaf, cardinality)
    marginal = torch.autograd.grad(differentiable_z, leaf)[0]
    assert torch.allclose(marginal, expected_marginal, atol=1e-6)


def test_zero_cardinality_head_is_exact_factorized_identity():
    field = _field().eval()
    values = _inputs()
    output = field(**values, return_marginals=True)
    assert torch.count_nonzero(field.cardinality_head.weight) == 0
    assert torch.count_nonzero(field.cardinality_head.bias) == 0
    assert torch.allclose(
        output.visible_role_marginals,
        values["visible_role_probs"],
        atol=1e-6,
    )
    assert torch.allclose(
        output.typed_null_role_marginals,
        values["typed_null_role_probs"],
        atol=1e-6,
    )

    expected_event = torch.zeros(2, 3)
    for human in range(2):
        rows = values["pair_agent_indices"] == human
        for action in range(3):
            visible_survival = torch.prod(
                1.0 - values["visible_role_probs"][rows, action].sum(-1)
            )
            valid = VALID_ROLES[action]
            typed_survival = torch.prod(
                1.0 - values["typed_null_role_probs"][human, action, valid]
            )
            expected_event[human, action] = 1.0 - (
                visible_survival * typed_survival
            )
    assert torch.allclose(output.event_probabilities, expected_event, atol=1e-6)


def test_registered_ablations_change_exactly_one_full_factor():
    full = asdict(GroundedRoleSetAblation.from_id("full"))
    expected = {
        "no_line_graph": "use_line_graph",
        "no_cardinality": "learn_cardinality",
        "hard_localization": "localization_target_mode",
        "no_typed_null": "use_typed_null_atoms",
        "posthoc_detector": "detector_score_mode",
    }
    for ablation_id, changed_field in expected.items():
        control = asdict(GroundedRoleSetAblation.from_id(ablation_id))
        changed = {
            name
            for name, value in control.items()
            if name != "ablation_id" and value != full[name]
        }
        assert changed == {changed_field}


def test_no_cardinality_control_is_fixed_factorized_identity():
    field = GroundedRoleSetField(
        GroundedRoleSetConfig(
            d_model=8,
            num_roles=2,
            max_cardinality=8,
            learn_cardinality=False,
        )
    ).train()
    assert not any(parameter.requires_grad for parameter in field.parameters())
    values = _inputs()
    output = field(**values, return_marginals=True)
    assert torch.allclose(
        output.visible_role_marginals,
        values["visible_role_probs"],
        atol=1e-6,
    )
    assert torch.allclose(
        output.typed_null_role_marginals,
        values["typed_null_role_probs"],
        atol=1e-6,
    )


def test_candidate_permutation_is_equivariant():
    field = _field().eval()
    values = _inputs()
    original = field(**values, return_marginals=True)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    permuted_values = dict(values)
    for key in ("visible_role_probs", "pair_agent_indices", "visible_action_mask"):
        permuted_values[key] = values[key][permutation]
    permuted = field(**permuted_values, return_marginals=True)
    assert torch.allclose(
        original.visible_role_marginals,
        permuted.visible_role_marginals[inverse],
        atol=1e-6,
    )
    assert torch.allclose(
        original.typed_null_role_marginals,
        permuted.typed_null_role_marginals,
        atol=1e-6,
    )
    assert torch.allclose(
        original.event_probabilities, permuted.event_probabilities, atol=1e-6
    )


def test_latent_localization_target_sums_alternatives_not_duplicates():
    field = _field().eval()
    event = field(**_inputs()).event(0, 0)
    group = RoleFillerTargetGroup(
        role_index=0, candidate_rows=torch.tensor([0, 1])
    )
    target_mass = field.target_log_mass(event, [group])
    assignments = []
    for selected in (0, 1):
        state = torch.zeros(event.num_candidates, dtype=torch.long)
        state[selected] = 1
        assignments.append(
            event.log_state_weights[
                torch.arange(event.num_candidates), state
            ].sum()
            + event.cardinality_energy[1]
        )
    expected = torch.logsumexp(torch.stack(assignments), dim=0)
    assert target_mass is not None
    assert torch.allclose(target_mass, expected, atol=1e-6)

    # The target is one latent filler, not the assignment that marks both
    # overlapping proposals as independent positive interactions.
    duplicate_state = torch.zeros(event.num_candidates, dtype=torch.long)
    duplicate_state[:2] = 1
    duplicate_mass = event.log_state_weights[
        torch.arange(event.num_candidates), duplicate_state
    ].sum() + event.cardinality_energy[2]
    assert not torch.allclose(target_mass, duplicate_mass)


def test_two_typed_null_roles_are_jointly_representable():
    field = _field().eval()
    event = field(**_inputs()).event(0, 1)
    first = event.typed_candidate_row(0)
    second = event.typed_candidate_row(1)
    groups = [
        RoleFillerTargetGroup(0, torch.tensor([first])),
        RoleFillerTargetGroup(1, torch.tensor([second])),
    ]
    target_mass = field.target_log_mass(event, groups)
    assert target_mass is not None and torch.isfinite(target_mass)
    state = torch.zeros(event.num_candidates, dtype=torch.long)
    state[first] = 1
    state[second] = 2
    expected = event.log_state_weights[
        torch.arange(event.num_candidates), state
    ].sum() + event.cardinality_energy[2]
    assert torch.allclose(target_mass, expected, atol=1e-6)


def test_one_candidate_cannot_fill_two_roles():
    field = _field().eval()
    event = field(**_inputs()).event(0, 1)
    groups = [
        RoleFillerTargetGroup(0, torch.tensor([0])),
        RoleFillerTargetGroup(1, torch.tensor([0])),
    ]
    assert field.target_log_mass(event, groups) is None


def test_distinct_candidates_can_fill_the_same_role_for_one_to_many_events():
    field = _field().eval()
    event = field(**_inputs()).event(0, 0)
    groups = [
        RoleFillerTargetGroup(0, torch.tensor([0])),
        RoleFillerTargetGroup(0, torch.tensor([1])),
    ]
    target = field.target_log_mass(event, groups)
    assert target is not None and torch.isfinite(target)
    state = torch.zeros(event.num_candidates, dtype=torch.long)
    state[0] = 1
    state[1] = 1
    expected = event.log_state_weights[
        torch.arange(event.num_candidates), state
    ].sum() + event.cardinality_energy[2]
    assert torch.allclose(target, expected, atol=1e-6)


def test_structured_likelihood_backpropagates_to_unaries_and_cardinality():
    field = _field().train()
    values = _inputs()
    visible = values["visible_role_probs"].clone().requires_grad_(True)
    typed = values["typed_null_role_probs"].clone().requires_grad_(True)
    output = field(
        **{
            **values,
            "visible_role_probs": visible,
            "typed_null_role_probs": typed,
        }
    )
    losses = []
    for event in output.events:
        if event.action_index == 0 and event.num_visible_candidates:
            target = field.target_log_mass(
                event,
                [RoleFillerTargetGroup(0, torch.tensor([0]))],
            )
        else:
            target = field.target_log_mass(event, [])
        assert target is not None
        losses.append(event.log_partition - target)
    loss = torch.stack(losses).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert visible.grad is not None and torch.isfinite(visible.grad).all()
    assert typed.grad is not None and torch.isfinite(typed.grad).all()
    assert field.cardinality_head.weight.grad is not None
    assert torch.count_nonzero(field.cardinality_head.weight.grad) > 0


def test_saturated_zero_support_keeps_exact_dp_gradients_finite():
    field = GroundedRoleSetField(
        GroundedRoleSetConfig(d_model=4, num_roles=2, max_cardinality=8)
    )
    visible = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 0.0]]], requires_grad=True
    )
    typed = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    output = field(
        visible_role_probs=visible,
        typed_null_role_probs=typed,
        pair_agent_indices=torch.tensor([0, 0]),
        event_states=torch.randn(1, 1, 4),
        visible_action_mask=torch.ones(2, 1, dtype=torch.bool),
        valid_role_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    event = output.event(0, 0)
    target = field.target_log_mass(
        event,
        [
            RoleFillerTargetGroup(0, torch.tensor([0])),
            RoleFillerTargetGroup(
                1, torch.tensor([event.typed_candidate_row(1)])
            ),
        ],
    )
    assert target is not None
    nll = event.log_partition - target
    loss = field.focal_transform_nll(nll, gamma=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert visible.grad is not None and torch.isfinite(visible.grad).all()
    assert typed.grad is not None and torch.isfinite(typed.grad).all()
    gradients = [
        parameter.grad
        for parameter in field.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_focal_transform_is_finite_at_the_exact_zero_limit():
    nll = torch.tensor([0.0, 1e-8, 0.2, 4.0], requires_grad=True)
    transformed = GroundedRoleSetField.focal_transform_nll(nll, gamma=0.1)
    expected_positive = nll[1:] * (1.0 - torch.exp(-nll[1:])).pow(0.1)
    assert transformed[0] == 0
    assert torch.allclose(transformed[1:], expected_positive, atol=1e-7)
    transformed.sum().backward()
    assert nll.grad is not None and torch.isfinite(nll.grad).all()
    assert nll.grad[0] == 0


def test_bfloat16_autocast_keeps_exact_dp_finite():
    field = _field().train()
    values = _inputs()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = field(**values)
        loss = torch.stack([event.log_partition for event in output.events]).mean()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [value.grad for value in field.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_empty_detected_agent_set_returns_device_consistent_empty_marginals():
    field = _field().eval()
    output = field(
        visible_role_probs=torch.empty(0, 3, 2),
        typed_null_role_probs=torch.empty(0, 3, 2),
        pair_agent_indices=torch.empty(0, dtype=torch.long),
        event_states=torch.empty(0, 3, 8),
        visible_action_mask=torch.empty(0, 3, dtype=torch.bool),
        valid_role_mask=VALID_ROLES,
        return_marginals=True,
    )
    assert output.events == ()
    assert output.visible_role_marginals.shape == (0, 3, 2)
    assert output.typed_null_role_marginals.shape == (0, 3, 2)
    assert output.event_probabilities.shape == (0, 3)
    assert output.visible_role_marginals.device == field.cardinality_head.weight.device


def test_function8_joint_event_is_exactly_representable_and_trainable():
    torch.manual_seed(812)
    field = GroundedRoleSetField(
        GroundedRoleSetConfig(d_model=16, num_roles=8, max_cardinality=12)
    )
    visible = torch.full((4, 2, 8), 0.02, requires_grad=True)
    typed = torch.full((2, 2, 8), 0.03, requires_grad=True)
    output = field(
        visible_role_probs=visible,
        typed_null_role_probs=typed,
        pair_agent_indices=torch.tensor([0, 0, 1, 1]),
        event_states=torch.randn(2, 2, 16),
        visible_action_mask=torch.ones(4, 2, dtype=torch.bool),
        valid_role_mask=torch.ones(2, 8, dtype=torch.bool),
        return_marginals=True,
    )
    assert output.visible_role_marginals.shape == (4, 2, 8)
    assert output.typed_null_role_marginals.shape == (2, 2, 8)
    assert torch.isfinite(output.visible_role_marginals).all()

    event = output.event(0, 0)
    groups = [
        RoleFillerTargetGroup(
            role,
            torch.tensor([event.typed_candidate_row(role)]),
        )
        for role in range(8)
    ]
    target = field.target_log_mass(event, groups)
    assert target is not None and torch.isfinite(target)
    loss = event.log_partition - target
    loss.backward()
    assert visible.grad is not None and torch.isfinite(visible.grad).all()
    assert typed.grad is not None and torch.isfinite(typed.grad).all()
    assert field.cardinality_head.weight.grad is not None
