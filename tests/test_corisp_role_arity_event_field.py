from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import (  # noqa: E402
    RoleArityEvent,
    RoleArityEventField,
    RoleArityEventFieldConfig,
    RoleFillerTargetGroup,
)


def _case(
    *,
    roles: int = 2,
    d_model: int = 12,
) -> tuple[RoleArityEventField, dict[str, torch.Tensor]]:
    torch.manual_seed(401)
    humans, actions, q = 2, 3, 5
    field = RoleArityEventField(
        RoleArityEventFieldConfig(
            d_model=d_model,
            num_roles=roles,
            max_cardinality=12,
            num_heads=3,
            arity_rank=8,
            dropout=0.0,
        )
    ).eval()
    visible = torch.rand(q, actions, roles)
    visible = 0.45 * visible / visible.sum(dim=-1, keepdim=True)
    return field, {
        "base_visible_role_probs": visible,
        "base_typed_null_role_probs": 0.25 * torch.rand(humans, actions, roles),
        "edge_states": torch.randn(q, actions, d_model),
        "null_edge_states": torch.randn(humans, actions, d_model),
        "event_states": torch.randn(humans, actions, d_model),
        "pair_agent_indices": torch.tensor([0, 0, 1, 1, 0]),
        "pair_entity_indices": torch.tensor([0, 1, 1, 2, 2]),
        "noun_anchors": torch.randn(q, d_model),
        "action_anchors": torch.randn(actions, d_model),
        "role_anchors": torch.randn(roles, d_model),
        "visible_action_mask": torch.ones(q, actions, dtype=torch.bool),
        "valid_role_mask": torch.ones(actions, roles, dtype=torch.bool),
    }


def _exhaustive(
    log_states: torch.Tensor,
    active_roles: torch.Tensor,
    energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = list(
        itertools.product(range(log_states.shape[1]), repeat=log_states.shape[0])
    )
    coefficient = energy.new_full(energy.shape, float("-inf"))
    values: list[torch.Tensor] = []
    valid_assignments: list[tuple[int, ...]] = []
    for assignment in assignments:
        state = torch.tensor(assignment)
        selected_roles = state[state > 0] - 1
        if any(
            int(role) not in active_roles.tolist()
            for role in selected_roles.tolist()
        ):
            continue
        counts = [
            int((selected_roles == int(role)).sum())
            for role in active_roles.tolist()
        ]
        cardinality = int((state > 0).sum())
        arity = RoleArityEventField.encode_arity_counts(counts)
        unary = log_states[torch.arange(len(state)), state].sum()
        coefficient[cardinality, arity] = torch.logaddexp(
            coefficient[cardinality, arity], unary
        )
        values.append(unary + energy[cardinality, arity])
        valid_assignments.append(assignment)
    stacked = torch.stack(values)
    log_z = torch.logsumexp(stacked, dim=0)
    probability = torch.softmax(stacked, dim=0)
    marginal = torch.zeros_like(log_states)
    for assignment, weight in zip(valid_assignments, probability):
        for row, state in enumerate(assignment):
            marginal[row, state] += weight
    return log_z, marginal, coefficient


def test_role_arity_dp_matches_exhaustive_partition_coefficients_and_marginals() -> None:
    log_states = torch.log(
        torch.tensor(
            [
                [0.35, 0.40, 0.25],
                [0.45, 0.15, 0.40],
                [0.30, 0.50, 0.20],
            ]
        )
    )
    active = torch.tensor([0, 1])
    energy = torch.linspace(-0.4, 0.7, 4 * 9).reshape(4, 9)
    expected_z, expected_marginal, expected_coefficient = _exhaustive(
        log_states, active, energy
    )
    actual_z, actual_coefficient = RoleArityEventField.log_arity_partition(
        log_states, active, energy
    )
    reachable = torch.isfinite(expected_coefficient)
    assert torch.equal(torch.isfinite(actual_coefficient), reachable)
    assert torch.allclose(
        actual_coefficient[reachable], expected_coefficient[reachable], atol=1e-6
    )
    assert torch.allclose(actual_z, expected_z, atol=1e-6)

    leaf = log_states.clone().requires_grad_(True)
    differentiable_z, _ = RoleArityEventField.log_arity_partition(
        leaf, active, energy
    )
    actual_marginal = torch.autograd.grad(differentiable_z, leaf)[0]
    assert torch.allclose(actual_marginal, expected_marginal, atol=1e-6)


def test_arity_state_distinguishes_single_and_repeated_fillers_per_role() -> None:
    assert RoleArityEventField.encode_arity_counts([0, 0]) == 0
    assert RoleArityEventField.encode_arity_counts([1, 0]) == 1
    assert RoleArityEventField.encode_arity_counts([2, 0]) == 2
    assert RoleArityEventField.encode_arity_counts([8, 0]) == 2
    assert RoleArityEventField.encode_arity_counts([1, 1]) == 4
    assert RoleArityEventField.encode_arity_counts([2, 1]) == 5
    assert RoleArityEventField.encode_arity_counts([1, 2]) == 7


def test_zero_gates_are_exact_factorized_identity() -> None:
    field, inputs = _case()
    output = field(**inputs, return_marginals=True)
    assert field.refinement_log_scale.item() == 0.0
    assert field.arity_log_scale.item() == 0.0
    assert torch.count_nonzero(field.cardinality_head.weight) == 0
    assert torch.allclose(
        output.visible_role_marginals,
        inputs["base_visible_role_probs"],
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        output.typed_null_role_marginals,
        inputs["base_typed_null_role_probs"],
        atol=2e-6,
        rtol=2e-6,
    )
    assert output.role_arity_probabilities is not None
    assert torch.allclose(
        output.role_arity_probabilities.sum(dim=-1),
        torch.ones(2, 3, 2),
        atol=1e-6,
    )


def test_typed_null_role_mask_can_disable_all_typed_candidates() -> None:
    field, inputs = _case(roles=1)
    typed_mask = torch.zeros_like(inputs["valid_role_mask"])
    output = field(
        **inputs,
        typed_null_role_mask=typed_mask,
        return_marginals=True,
    )
    assert all(event.typed_role_indices.numel() == 0 for event in output.events)
    assert output.typed_null_role_marginals is not None
    assert torch.count_nonzero(output.typed_null_role_marginals) == 0
    assert torch.allclose(
        output.visible_role_marginals,
        inputs["base_visible_role_probs"],
        atol=2e-6,
        rtol=2e-6,
    )


def test_omitting_deterministic_null_events_is_exact() -> None:
    field, inputs = _case(roles=1)
    field.refinement_log_scale.data.fill_(0.2)
    field.arity_log_scale.data.fill_(0.2)
    inputs["visible_action_mask"][:, 2] = False
    typed_mask = torch.zeros_like(inputs["valid_role_mask"])
    dense = field(
        **inputs,
        typed_null_role_mask=typed_mask,
        return_marginals=True,
    )
    sparse = field(
        **inputs,
        typed_null_role_mask=typed_mask,
        omit_deterministic_null_events=True,
        return_marginals=True,
    )
    assert len(dense.events) == 6
    assert len(sparse.events) == 4
    for name in (
        "visible_role_marginals",
        "typed_null_role_marginals",
        "event_probabilities",
        "cardinality_expectations",
        "role_arity_probabilities",
        "capped_role_count_expectations",
    ):
        assert torch.allclose(
            getattr(sparse, name), getattr(dense, name), atol=3e-6, rtol=3e-5
        )
    assert sparse.event(0, 0).action_index == 0
    try:
        sparse.event(0, 2)
    except KeyError:
        pass
    else:
        raise AssertionError("An omitted deterministic-null event was materialized.")


def test_typed_null_role_mask_must_be_a_valid_role_subset() -> None:
    field, inputs = _case()
    inputs["valid_role_mask"][0, 1] = False
    invalid = torch.ones_like(inputs["valid_role_mask"])
    try:
        field(**inputs, typed_null_role_mask=invalid)
    except ValueError as error:
        assert "subset" in str(error)
    else:
        raise AssertionError("An invalid typed-null role mask was accepted.")


def test_candidate_permutation_is_equivariant() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.3)
    field.arity_log_scale.data.fill_(0.25)
    reference = field(**inputs, return_marginals=True)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.argsort(permutation)
    permuted = dict(inputs)
    for name in (
        "base_visible_role_probs",
        "edge_states",
        "pair_agent_indices",
        "pair_entity_indices",
        "noun_anchors",
        "visible_action_mask",
    ):
        permuted[name] = inputs[name][permutation]
    result = field(**permuted, return_marginals=True)
    assert torch.allclose(
        result.visible_role_marginals[inverse],
        reference.visible_role_marginals,
        atol=3e-6,
        rtol=3e-5,
    )
    assert torch.allclose(
        result.role_arity_probabilities,
        reference.role_arity_probabilities,
        atol=3e-6,
        rtol=3e-5,
    )


def test_role_axis_permutation_preserves_semantics() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.3)
    field.arity_log_scale.data.fill_(0.25)
    reference = field(**inputs, return_marginals=True)
    permutation = torch.tensor([1, 0])
    permuted = dict(inputs)
    permuted["base_visible_role_probs"] = inputs[
        "base_visible_role_probs"
    ][..., permutation]
    permuted["base_typed_null_role_probs"] = inputs[
        "base_typed_null_role_probs"
    ][..., permutation]
    permuted["role_anchors"] = inputs["role_anchors"][permutation]
    permuted["valid_role_mask"] = inputs["valid_role_mask"][:, permutation]
    result = field(**permuted, return_marginals=True)
    assert torch.allclose(
        result.visible_role_marginals[..., permutation],
        reference.visible_role_marginals,
        atol=3e-6,
        rtol=3e-5,
    )
    assert torch.allclose(
        result.role_arity_probabilities[..., permutation, :],
        reference.role_arity_probabilities,
        atol=3e-6,
        rtol=3e-5,
    )


def test_same_role_gold_permutations_are_quotiented_once() -> None:
    log_states = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    active = torch.tensor([0])
    energy = torch.zeros(3, 3)
    log_z, coefficient = RoleArityEventField.log_arity_partition(
        log_states, active, energy
    )
    event = RoleArityEvent(
        agent_index=0,
        action_index=0,
        pair_indices=torch.tensor([0, 1]),
        typed_role_indices=torch.empty(0, dtype=torch.long),
        active_role_indices=active,
        log_state_weights=log_states,
        composition_energy=energy,
        log_composition_coefficients=coefficient,
        log_partition=log_z,
    )
    groups = [
        RoleFillerTargetGroup(0, torch.tensor([0, 1])),
        RoleFillerTargetGroup(0, torch.tensor([0, 1])),
    ]
    target = RoleArityEventField.target_log_mass(event, groups)
    assert target is not None
    # Both Gold-to-proposal bijections induce the same predicted multiset.
    expected = log_states[:, 1].sum() + energy[2, 2]
    assert torch.allclose(target, expected, atol=1e-7)


def test_distinct_predicted_states_remain_marginalized() -> None:
    log_states = torch.log(
        torch.tensor(
            [
                [0.4, 0.3, 0.3],
                [0.5, 0.2, 0.3],
                [0.6, 0.2, 0.2],
            ]
        )
    )
    active = torch.tensor([0, 1])
    energy = torch.zeros(4, 9)
    log_z, coefficient = RoleArityEventField.log_arity_partition(
        log_states, active, energy
    )
    event = RoleArityEvent(
        agent_index=0,
        action_index=0,
        pair_indices=torch.arange(3),
        typed_role_indices=torch.empty(0, dtype=torch.long),
        active_role_indices=active,
        log_state_weights=log_states,
        composition_energy=energy,
        log_composition_coefficients=coefficient,
        log_partition=log_z,
    )
    groups = [
        RoleFillerTargetGroup(0, torch.tensor([0, 1])),
        RoleFillerTargetGroup(1, torch.tensor([1, 2])),
    ]
    target = RoleArityEventField.target_log_mass(event, groups)
    assert target is not None
    states = (
        torch.tensor([1, 2, 0]),
        torch.tensor([1, 0, 2]),
        torch.tensor([0, 1, 2]),
    )
    values = [
        log_states[torch.arange(3), state].sum() + energy[2, 4]
        for state in states
    ]
    assert torch.allclose(target, torch.logsumexp(torch.stack(values), 0), atol=1e-7)


def test_relational_and_arity_parameters_receive_joint_likelihood_gradients() -> None:
    field, inputs = _case()
    field.train()
    field.refinement_log_scale.data.fill_(0.2)
    field.arity_log_scale.data.fill_(0.2)
    output = field(**inputs)
    losses = []
    for event in output.events:
        groups = []
        if event.action_index == 0 and event.num_visible_candidates:
            groups = [RoleFillerTargetGroup(0, torch.tensor([0]))]
        target = field.target_log_mass(event, groups)
        assert target is not None
        losses.append(event.log_partition - target)
    loss = torch.stack(losses).mean()
    loss.backward()
    expected = (
        field.refinement_log_scale,
        field.query_projection[1].weight,
        field.value_projection[1].weight,
        field.cardinality_head.weight,
        field.arity_log_scale,
        field.arity_role_projection[1].weight,
        field.arity_count_embeddings,
        field.arity_set_projection[1].weight,
    )
    for parameter in expected:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_function8_exact_forward_backward_is_finite() -> None:
    torch.manual_seed(503)
    roles, d_model = 8, 16
    field = RoleArityEventField(
        RoleArityEventFieldConfig(
            d_model=d_model,
            num_roles=roles,
            max_cardinality=10,
            num_heads=4,
            arity_rank=16,
            dropout=0.0,
        )
    ).train()
    field.refinement_log_scale.data.fill_(0.1)
    field.arity_log_scale.data.fill_(0.1)
    visible = torch.full((2, 1, roles), 0.01, requires_grad=True)
    typed = torch.full((1, 1, roles), 0.02, requires_grad=True)
    output = field(
        base_visible_role_probs=visible,
        base_typed_null_role_probs=typed,
        edge_states=torch.randn(2, 1, d_model),
        null_edge_states=torch.randn(1, 1, d_model),
        event_states=torch.randn(1, 1, d_model),
        pair_agent_indices=torch.tensor([0, 0]),
        pair_entity_indices=torch.tensor([0, 1]),
        noun_anchors=torch.randn(2, d_model),
        action_anchors=torch.randn(1, d_model),
        role_anchors=torch.randn(roles, d_model),
        visible_action_mask=torch.ones(2, 1, dtype=torch.bool),
        valid_role_mask=torch.ones(1, roles, dtype=torch.bool),
    )
    event = output.event(0, 0)
    assert event.num_arity_states == 6561
    groups = [
        RoleFillerTargetGroup(
            role, torch.tensor([event.typed_candidate_row(role)])
        )
        for role in range(roles)
    ]
    target = field.target_log_mass(event, groups)
    assert target is not None and torch.isfinite(target)
    loss = event.log_partition - target
    loss.backward()
    assert visible.grad is not None and torch.isfinite(visible.grad).all()
    assert typed.grad is not None and torch.isfinite(typed.grad).all()


def test_cpu_bfloat16_forward_and_backward_are_finite() -> None:
    field, inputs = _case(d_model=12)
    field.train()
    field.refinement_log_scale.data.fill_(0.2)
    field.arity_log_scale.data.fill_(0.2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = field(**inputs)
        losses = []
        for event in output.events:
            groups = []
            if event.action_index == 0 and event.num_visible_candidates:
                groups = [RoleFillerTargetGroup(0, torch.tensor([0]))]
            target = field.target_log_mass(event, groups)
            assert target is not None
            losses.append(event.log_partition - target)
        loss = torch.stack(losses).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in field.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_empty_detected_agent_set_returns_consistent_empty_marginals() -> None:
    field, _ = _case()
    output = field(
        base_visible_role_probs=torch.empty(0, 3, 2),
        base_typed_null_role_probs=torch.empty(0, 3, 2),
        edge_states=torch.empty(0, 3, 12),
        null_edge_states=torch.empty(0, 3, 12),
        event_states=torch.empty(0, 3, 12),
        pair_agent_indices=torch.empty(0, dtype=torch.long),
        pair_entity_indices=torch.empty(0, dtype=torch.long),
        noun_anchors=torch.empty(0, 12),
        action_anchors=torch.randn(3, 12),
        role_anchors=torch.randn(2, 12),
        visible_action_mask=torch.empty(0, 3, dtype=torch.bool),
        valid_role_mask=torch.ones(3, 2, dtype=torch.bool),
        return_marginals=True,
    )
    assert output.events == ()
    assert output.visible_role_marginals is not None
    assert output.visible_role_marginals.shape == (0, 3, 2)
    assert output.role_arity_probabilities is not None
    assert output.role_arity_probabilities.shape == (0, 3, 2, 3)
    assert output.capped_role_count_expectations is not None
    assert output.capped_role_count_expectations.shape == (0, 3, 2)
