from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import (  # noqa: E402
    GroundedRoleSetConfig,
    GroundedRoleSetField,
    ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS,
    RoleFillerEventField,
    RoleFillerEventFieldConfig,
    RoleFillerTargetGroup,
)


def _case(
    *,
    roles: int = 2,
    d_model: int = 12,
) -> tuple[RoleFillerEventField, dict[str, torch.Tensor]]:
    torch.manual_seed(101)
    humans, actions, q = 2, 3, 5
    field = RoleFillerEventField(
        RoleFillerEventFieldConfig(
            d_model=d_model,
            num_roles=roles,
            max_cardinality=12,
            num_heads=3,
            signature_rank=8,
            dropout=0.0,
        )
    ).eval()
    valid_role_mask = torch.ones(actions, roles, dtype=torch.bool)
    visible = torch.rand(q, actions, roles)
    visible = 0.45 * visible / visible.sum(dim=-1, keepdim=True)
    typed = 0.25 * torch.rand(humans, actions, roles)
    return field, {
        "base_visible_role_probs": visible,
        "base_typed_null_role_probs": typed,
        "edge_states": torch.randn(q, actions, d_model),
        "null_edge_states": torch.randn(humans, actions, d_model),
        "event_states": torch.randn(humans, actions, d_model),
        "pair_agent_indices": torch.tensor([0, 0, 1, 1, 0]),
        "pair_entity_indices": torch.tensor([0, 1, 1, 2, 2]),
        "noun_anchors": torch.randn(q, d_model),
        "action_anchors": torch.randn(actions, d_model),
        "role_anchors": torch.randn(roles, d_model),
        "visible_action_mask": torch.ones(q, actions, dtype=torch.bool),
        "valid_role_mask": valid_role_mask,
    }


def _exhaustive(log_states: torch.Tensor, energy: torch.Tensor):
    assignments = list(
        itertools.product(range(log_states.shape[1]), repeat=log_states.shape[0])
    )
    values = []
    for assignment in assignments:
        state = torch.tensor(assignment)
        cardinality = int((state > 0).sum())
        signature = 0
        for selected in state[state > 0].tolist():
            signature |= 1 << (int(selected) - 1)
        value = log_states[torch.arange(len(state)), state].sum()
        values.append(value + energy[cardinality, signature])
    values = torch.stack(values)
    log_z = torch.logsumexp(values, dim=0)
    probability = torch.softmax(values, dim=0)
    marginal = torch.zeros_like(log_states)
    for assignment, weight in zip(assignments, probability):
        for row, state in enumerate(assignment):
            marginal[row, state] += weight
    return log_z, marginal


def test_role_signature_dp_matches_exhaustive_partition_and_marginals() -> None:
    log_states = torch.log(
        torch.tensor(
            [
                [0.50, 0.30, 0.20],
                [0.55, 0.35, 0.10],
                [0.45, 0.15, 0.40],
            ]
        )
    )
    energy = torch.tensor(
        [
            [0.0, -0.2, 0.1, 0.3],
            [0.2, 0.4, -0.1, 0.5],
            [-0.3, 0.2, 0.6, -0.2],
            [-0.5, 0.1, -0.4, 0.7],
        ]
    )
    expected_z, expected_marginal = _exhaustive(log_states, energy)
    actual_z, coefficient = RoleFillerEventField.log_partition(log_states, energy)
    assert coefficient.shape == energy.shape
    assert torch.allclose(actual_z, expected_z, atol=1e-6)

    leaf = log_states.clone().requires_grad_(True)
    differentiable_z, _ = RoleFillerEventField.log_partition(leaf, energy)
    actual_marginal = torch.autograd.grad(differentiable_z, leaf)[0]
    assert torch.allclose(actual_marginal, expected_marginal, atol=1e-6)


def test_zero_gates_are_exact_factorized_identity() -> None:
    field, inputs = _case()
    output = field(**inputs, return_marginals=True)
    assert field.refinement_log_scale.item() == 0.0
    assert field.signature_log_scale.item() == 0.0
    assert torch.count_nonzero(field.cardinality_head.weight) == 0
    assert torch.allclose(
        output.visible_role_marginals,
        inputs["base_visible_role_probs"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        output.typed_null_role_marginals,
        inputs["base_typed_null_role_probs"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert output.role_signature_probabilities is not None
    assert torch.allclose(
        output.role_signature_probabilities.sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )


def test_unary_only_is_exact_base_probability_identity() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.7)
    field.signature_log_scale.data.fill_(-0.6)
    torch.nn.init.normal_(field.cardinality_head.weight, std=0.2)
    torch.nn.init.normal_(field.cardinality_head.bias, std=0.2)
    output = field(
        **inputs,
        return_marginals=True,
        intervention="unary_only",
    )
    assert output.refinement_scale.item() == 0.0
    assert output.signature_scale.item() == 0.0
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


def test_interventions_are_validated_and_inference_only() -> None:
    field, inputs = _case()
    with torch.no_grad():
        try:
            field(**inputs, intervention="not_an_intervention")
        except ValueError as error:
            assert "Unsupported role-filler intervention" in str(error)
        else:
            raise AssertionError("Unknown intervention was accepted.")
    field.train()
    try:
        field(**inputs, intervention="no_signature")
    except ValueError as error:
        assert "inference-only" in str(error)
    else:
        raise AssertionError("A training-time intervention was accepted.")


def test_structural_ablations_are_trainable_and_exclusive_with_interventions() -> None:
    for structural_ablation in ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS:
        field, inputs = _case()
        field.train()
        result = field(
            **inputs,
            return_marginals=True,
            structural_ablation=structural_ablation,
        )
        assert torch.isfinite(result.visible_role_marginals).all()
        assert torch.isfinite(result.typed_null_role_marginals).all()

    field, inputs = _case()
    try:
        field(**inputs, structural_ablation="not_an_ablation")
    except ValueError as error:
        assert "structural ablation" in str(error)
    else:
        raise AssertionError("Unknown structural ablation was accepted.")
    try:
        field(
            **inputs,
            intervention="no_signature",
            structural_ablation="cardinality_only",
        )
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("Intervention and structural ablation were combined.")


def test_structural_ablation_identities_are_exact() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.7)
    field.signature_log_scale.data.fill_(0.6)
    torch.nn.init.normal_(field.cardinality_head.weight, std=0.2)
    torch.nn.init.normal_(field.cardinality_head.bias, std=0.2)

    unary = field(
        **inputs,
        return_marginals=True,
        structural_ablation="unary_only",
    )
    assert torch.allclose(
        unary.visible_role_marginals,
        inputs["base_visible_role_probs"],
        atol=2e-6,
        rtol=2e-6,
    )
    assert unary.refinement_scale.item() == 0.0
    assert unary.signature_scale.item() == 0.0

    cardinality = field(
        **inputs,
        return_marginals=True,
        structural_ablation="cardinality_only",
    )
    assert cardinality.refinement_scale.item() == 0.0
    assert cardinality.signature_scale.item() == 0.0
    assert not torch.allclose(
        cardinality.visible_role_marginals,
        unary.visible_role_marginals,
        atol=1e-8,
        rtol=1e-7,
    )


def test_factor_and_relation_interventions_are_finite_and_nontrivial() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.5)
    field.signature_log_scale.data.fill_(0.4)
    torch.nn.init.normal_(field.cardinality_head.weight, std=0.1)
    torch.nn.init.normal_(field.cardinality_head.bias, std=0.1)
    reference = field(**inputs, return_marginals=True)
    for intervention in (
        "no_unary_refinement",
        "no_cardinality",
        "no_signature",
        "no_event_relation",
        "no_pair_relation",
        "no_entity_relation",
        "no_all_relations",
        "role_collapsed_routing",
    ):
        result = field(
            **inputs,
            return_marginals=True,
            intervention=intervention,
        )
        assert torch.isfinite(result.visible_role_marginals).all()
        assert torch.isfinite(result.typed_null_role_marginals).all()
        assert not torch.allclose(
            result.visible_role_marginals,
            reference.visible_role_marginals,
            atol=1e-8,
            rtol=1e-7,
        )


def test_disabled_v10_factors_exactly_reduce_to_v8_cardinality_field() -> None:
    field, inputs = _case()
    torch.manual_seed(211)
    torch.nn.init.normal_(field.cardinality_head.weight, std=0.04)
    torch.nn.init.normal_(field.cardinality_head.bias, std=0.03)

    baseline = GroundedRoleSetField(
        GroundedRoleSetConfig(
            d_model=field.cfg.d_model,
            num_roles=field.cfg.num_roles,
            max_cardinality=field.cfg.max_cardinality,
        )
    ).eval()
    baseline.cardinality_norm.load_state_dict(field.cardinality_norm.state_dict())
    baseline.cardinality_head.load_state_dict(field.cardinality_head.state_dict())

    actual = field(**inputs, return_marginals=True)
    expected = baseline(
        visible_role_probs=inputs["base_visible_role_probs"],
        typed_null_role_probs=inputs["base_typed_null_role_probs"],
        pair_agent_indices=inputs["pair_agent_indices"],
        event_states=inputs["event_states"],
        visible_action_mask=inputs["visible_action_mask"],
        valid_role_mask=inputs["valid_role_mask"],
        return_marginals=True,
    )

    for actual_event, expected_event in zip(actual.events, expected.events):
        assert torch.allclose(
            actual_event.log_partition,
            expected_event.log_partition,
            atol=3e-6,
            rtol=3e-6,
        )
    assert torch.allclose(
        actual.visible_role_marginals,
        expected.visible_role_marginals,
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.allclose(
        actual.typed_null_role_marginals,
        expected.typed_null_role_marginals,
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.allclose(
        actual.event_probabilities,
        expected.event_probabilities,
        atol=3e-6,
        rtol=3e-6,
    )
    assert torch.allclose(
        actual.cardinality_expectations,
        expected.cardinality_expectations,
        atol=3e-6,
        rtol=3e-6,
    )


def test_candidate_permutation_is_equivariant() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.35)
    field.signature_log_scale.data.fill_(0.25)
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
        atol=2e-6,
        rtol=2e-5,
    )
    assert torch.allclose(
        result.typed_null_role_marginals,
        reference.typed_null_role_marginals,
        atol=2e-6,
        rtol=2e-5,
    )
    assert torch.allclose(
        result.role_signature_probabilities,
        reference.role_signature_probabilities,
        atol=2e-6,
        rtol=2e-5,
    )


def test_role_axis_is_semantically_equivariant_not_a_class_table() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.35)
    field.signature_log_scale.data.fill_(0.25)
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
        atol=2e-6,
        rtol=2e-5,
    )
    assert torch.allclose(
        result.typed_null_role_marginals[..., permutation],
        reference.typed_null_role_marginals,
        atol=2e-6,
        rtol=2e-5,
    )


def test_peer_role_identity_is_not_collapsed_to_interaction_mass() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.5)
    first = dict(inputs)
    second = dict(inputs)
    first_visible = inputs["base_visible_role_probs"].clone()
    second_visible = first_visible.clone()
    # Candidate 1 belongs to the same person-event as candidate 0.  Swap its
    # role while preserving exactly the same total positive probability.
    first_visible[1, 0] = torch.tensor([0.35, 0.05])
    second_visible[1, 0] = torch.tensor([0.05, 0.35])
    first["base_visible_role_probs"] = first_visible
    second["base_visible_role_probs"] = second_visible
    first_output = field(**first, return_marginals=True)
    second_output = field(**second, return_marginals=True)
    assert torch.equal(first_visible[0, 0], second_visible[0, 0])
    assert first_visible[1, 0].sum() == second_visible[1, 0].sum()
    assert not torch.allclose(
        first_output.visible_role_marginals[0, 0],
        second_output.visible_role_marginals[0, 0],
        atol=1e-7,
        rtol=1e-6,
    )


def test_same_pair_relation_transmits_cross_predicate_evidence() -> None:
    field, inputs = _case()
    field.refinement_log_scale.data.fill_(0.5)
    first = field(**inputs, return_marginals=True)
    changed = dict(inputs)
    changed["edge_states"] = inputs["edge_states"].clone()
    changed["edge_states"][0, 1, 0] += 4.0
    second = field(**changed, return_marginals=True)
    assert not torch.allclose(
        first.visible_role_marginals[0, 0],
        second.visible_role_marginals[0, 0],
        atol=1e-7,
        rtol=1e-6,
    )


def test_signature_factor_distinguishes_equal_cardinality_role_sets() -> None:
    field, inputs = _case()
    field.signature_log_scale.data.fill_(0.5)
    _, _, role_context, _ = field._relational_unaries(**inputs)
    energy, scale = field._composition_energy(inputs["event_states"], role_context)
    assert scale.abs().item() > 0
    assert not torch.allclose(energy[..., 1, 1], energy[..., 1, 2])
    assert torch.allclose(energy[..., 0, 0], torch.zeros_like(energy[..., 0, 0]))


def test_target_mass_uses_role_signature_and_distinct_assignments() -> None:
    field, inputs = _case()
    field.signature_log_scale.data.fill_(0.4)
    event = field(**inputs).event(0, 0)
    groups = [
        RoleFillerTargetGroup(0, torch.tensor([0])),
        RoleFillerTargetGroup(1, torch.tensor([1])),
    ]
    target = field.target_log_mass(event, groups)
    assert target is not None
    state = torch.zeros(event.num_candidates, dtype=torch.long)
    state[0] = 1
    state[1] = 2
    expected = event.log_state_weights[
        torch.arange(event.num_candidates), state
    ].sum() + event.composition_energy[2, 3]
    assert torch.allclose(target, expected, atol=1e-6)

    impossible = [
        RoleFillerTargetGroup(0, torch.tensor([0])),
        RoleFillerTargetGroup(1, torch.tensor([0])),
    ]
    assert field.target_log_mass(event, impossible) is None


def test_relational_and_signature_parameters_receive_joint_likelihood_gradients() -> None:
    field, inputs = _case()
    field.train()
    field.refinement_log_scale.data.fill_(0.2)
    field.signature_log_scale.data.fill_(0.2)
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
    expected = (
        field.refinement_log_scale,
        field.query_projection[1].weight,
        field.value_projection[1].weight,
        field.cardinality_head.weight,
        field.signature_log_scale,
        field.signature_atom_projection[1].weight,
        field.signature_set_projection[1].weight,
    )
    for parameter in expected:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_function8_event_is_exact_finite_and_trainable() -> None:
    torch.manual_seed(303)
    roles, d_model = 8, 16
    field = RoleFillerEventField(
        RoleFillerEventFieldConfig(
            d_model=d_model,
            num_roles=roles,
            max_cardinality=10,
            num_heads=4,
            signature_rank=16,
            dropout=0.0,
        )
    ).train()
    field.refinement_log_scale.data.fill_(0.1)
    field.signature_log_scale.data.fill_(0.1)
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
        return_marginals=True,
    )
    assert output.role_signature_probabilities.shape == (1, 1, 256)
    assert torch.isfinite(output.role_signature_probabilities).all()
    event = output.event(0, 0)
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


def test_unreachable_signature_paths_keep_exact_gradients_finite() -> None:
    log_states = torch.log(
        torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5],
                [1.0, 0.0, 0.0],
            ],
            requires_grad=True,
        )
    )
    energy = torch.zeros(4, 4, requires_grad=True)
    log_z, coefficient = RoleFillerEventField.log_partition(log_states, energy)
    assert torch.isfinite(log_z)
    assert torch.isneginf(coefficient).any()
    log_z.backward()
    assert log_states.grad_fn is not None
    probability = log_states.detach().exp().requires_grad_(True)
    log_probability = RoleFillerEventField._log_probability(probability)
    value, _ = RoleFillerEventField.log_partition(log_probability, energy.detach())
    value.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_cpu_bfloat16_forward_and_backward_are_finite() -> None:
    field, inputs = _case(d_model=12)
    field.train()
    field.refinement_log_scale.data.fill_(0.2)
    field.signature_log_scale.data.fill_(0.2)
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
    assert output.visible_role_marginals.shape == (0, 3, 2)
    assert output.role_signature_probabilities.shape == (0, 3, 4)
