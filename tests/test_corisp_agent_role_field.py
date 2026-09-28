import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import (  # noqa: E402
    PairPacket,
    CoRISPAgentRoleField,
    CoRISPAgentRoleFieldConfig,
)


def _packet() -> PairPacket:
    torch.manual_seed(3)
    entity_boxes = torch.tensor(
        [
            [
                [0.05, 0.05, 0.35, 0.90],
                [0.55, 0.05, 0.90, 0.90],
                [0.35, 0.35, 0.55, 0.65],
                [0.70, 0.35, 0.85, 0.65],
            ]
        ]
    )
    subjects = torch.tensor([[0, 0, 1, 1]])
    objects = torch.tensor([[2, 1, 2, 3]])
    return PairPacket(
        pair_feats=torch.randn(1, 4, 8),
        dense_tokens=torch.randn(1, 8, 3, 3),
        subject_boxes=entity_boxes[:, subjects[0]],
        object_boxes=entity_boxes[:, objects[0]],
        entity_boxes=entity_boxes,
        entity_feats=torch.randn(1, 4, 8),
        entity_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        subject_indices=subjects,
        object_indices=objects,
        entity_labels=torch.tensor([[0, 0, 2, 3]]),
    )


def _model(
    object_action_mask: torch.Tensor | None = None,
    *,
    enable_typed_null_fillers: bool = True,
) -> CoRISPAgentRoleField:
    torch.manual_seed(5)
    return CoRISPAgentRoleField(
        CoRISPAgentRoleFieldConfig(
            d_model=8,
            visual_dim=6,
            semantic_dim=10,
            num_actions=3,
            roles=("target", "instrument"),
            num_object_classes=4,
            inference_steps=2,
            ffn_dim=16,
            dropout=0.0,
            enable_typed_null_fillers=enable_typed_null_fillers,
        ),
        action_prototypes=torch.randn(3, 10),
        object_prototypes=torch.randn(4, 10),
        valid_role_mask=torch.tensor(
            [[True, False], [True, True], [False, True]]
        ),
        object_action_mask=object_action_mask,
    )


def _inputs():
    torch.manual_seed(9)
    return torch.randn(1, 6, 4, 4), torch.randn(1, 10), torch.randn(1, 2, 6)


def test_identity_initialization_preserves_host_marginal_exactly():
    model = _model().eval()
    packet = _packet()
    visual, global_semantic, aligned = _inputs()
    host = torch.randn(1, packet.num_pairs, 3)
    null_host = torch.randn(1, 4, 3)
    output = model(
        packet,
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned,
        host_logits=host,
        null_host_logits=null_host,
    )
    assert model.residual_scale.item() == 0.0
    assert torch.allclose(output["hoi_probs"], host.sigmoid(), atol=1e-6)
    human = packet.entity_labels.eq(0)[..., None]
    expected_null = null_host.sigmoid() * human
    assert torch.allclose(output["null_hoi_probs"], expected_null, atol=1e-6)
    typed_null = output["null_role_probs"]
    typed_or = 1.0 - torch.prod(1.0 - typed_null, dim=-1)
    assert torch.allclose(typed_or, output["null_hoi_probs"], atol=1e-6)
    # Action 1 admits both roles.  Its two absent-role slots coexist rather
    # than competing in a categorical softmax.
    assert (typed_null[0, :2, 1] > 0).all()
    assert (
        typed_null[0, :2, 1].sum(dim=-1)
        > output["null_hoi_probs"][0, :2, 1]
    ).all()


def test_joint_states_normalize_and_have_real_agent_cardinality():
    model = _model().eval()
    visual, global_semantic, aligned = _inputs()
    output = model(
        _packet(),
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned,
    )
    visible_total = output["no_interaction_probs"] + output[
        "joint_role_probs"
    ].sum(dim=-1)
    assert torch.allclose(visible_total, torch.ones_like(visible_total), atol=1e-6)
    assert output["agent_entity_indices"].shape == (1, 2)
    assert output["event_states"].shape == (1, 2, 3, 8)
    assert output["event_probs"].shape == (1, 2, 3)
    names = [name for name, _ in model.named_parameters()]
    assert not any("event_queries" in name or "event_slots" in name for name in names)


def test_heir_default_disables_vcoco_only_typed_null_fillers():
    model = _model(enable_typed_null_fillers=False).eval()
    visual, global_semantic, aligned = _inputs()
    output = model(
        _packet(),
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned,
    )
    assert output["null_role_probs"].eq(0).all()
    assert output["null_hoi_probs"].eq(0).all()
    assert output["null_no_interaction_probs"].eq(1).all()


def test_ontology_illegal_noun_actions_cannot_enter_field_messages():
    legal = torch.ones(4, 3, dtype=torch.bool)
    legal[2, 0] = False
    model = _model(legal).eval()
    packet = _packet()
    visual, global_semantic, aligned = _inputs()
    output = model(
        packet,
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned,
        host_logits=torch.full((1, 4, 3), 20.0),
    )
    object_labels = packet.entity_labels[0, packet.object_indices[0]]
    illegal_pairs = object_labels.eq(2)
    assert output["visible_action_mask"][0, illegal_pairs, 0].eq(False).all()
    assert output["hoi_probs"][0, illegal_pairs, 0].eq(0).all()
    assert output["no_interaction_probs"][0, illegal_pairs, 0].eq(1).all()


def test_joint_softmax_stays_finite_fp32_with_bfloat16_extremes():
    model = _model().train()
    model.residual_scale.data.fill_(0.5)
    host = (torch.randn(2, 3) * 80.0).to(torch.bfloat16).requires_grad_(True)
    residual = (
        torch.randn(2, 3, 2) * 80.0
    ).to(torch.bfloat16).requires_grad_(True)
    interaction = (
        torch.randn(2, 3) * 80.0
    ).to(torch.bfloat16).requires_grad_(True)
    no_interaction, role = model._joint_state(host, residual, interaction)
    assert no_interaction.dtype == torch.float32
    assert role.dtype == torch.float32
    assert torch.isfinite(no_interaction).all()
    assert torch.isfinite(role).all()
    assert torch.isfinite(model._probability_logit(role)).all()
    (no_interaction.square().mean() + role.square().mean()).backward()
    assert host.grad is not None and torch.isfinite(host.grad).all()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()
    assert interaction.grad is not None and torch.isfinite(interaction.grad).all()


def _permuted_packet(packet: PairPacket):
    entity_permutation = torch.tensor([2, 0, 3, 1])
    old_to_new = torch.empty_like(entity_permutation)
    old_to_new[entity_permutation] = torch.arange(entity_permutation.numel())
    pair_permutation = torch.tensor([2, 0, 3, 1])
    inverse_pair = torch.empty_like(pair_permutation)
    inverse_pair[pair_permutation] = torch.arange(pair_permutation.numel())
    subject = old_to_new[packet.subject_indices[0, pair_permutation]]
    objects = old_to_new[packet.object_indices[0, pair_permutation]]
    entity_boxes = packet.entity_boxes[:, entity_permutation]
    permuted = PairPacket(
        pair_feats=packet.pair_feats[:, pair_permutation],
        dense_tokens=packet.dense_tokens,
        subject_boxes=entity_boxes[:, subject],
        object_boxes=entity_boxes[:, objects],
        entity_boxes=entity_boxes,
        entity_feats=packet.entity_feats[:, entity_permutation],
        entity_valid_mask=packet.entity_valid_mask[:, entity_permutation],
        subject_indices=subject[None],
        object_indices=objects[None],
        entity_labels=packet.entity_labels[:, entity_permutation],
    )
    # New human order is old identity 0 followed by old identity 1 here.
    return permuted, inverse_pair


def test_entity_and_pair_permutations_preserve_edge_predictions():
    model = _model().eval()
    model.residual_scale.data.fill_(0.4)
    packet = _packet()
    permuted, inverse_pair = _permuted_packet(packet)
    visual, global_semantic, aligned = _inputs()
    first = model(
        packet,
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned,
    )
    second = model(
        permuted,
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned,
    )
    assert torch.allclose(
        first["joint_role_probs"],
        second["joint_role_probs"][:, inverse_pair],
        atol=2e-6,
        rtol=2e-5,
    )


def test_open_field_backpropagates_through_shared_role_evidence():
    model = _model().train()
    model.residual_scale.data.fill_(0.2)
    visual, global_semantic, aligned = _inputs()
    output = model(
        _packet(),
        visual_grid=visual,
        global_semantic=global_semantic,
        aligned_agent_tokens=aligned.requires_grad_(True),
    )
    loss = output["joint_role_logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.role_residual.weight.grad is not None
    assert model.interaction_residual.weight.grad is not None
    assert model.agent_query_projection[1].weight.grad is None
    assert aligned.grad is not None
    assert torch.isfinite(aligned.grad).all()


def test_full_field_forward_backward_is_bfloat16_autocast_safe():
    model = _model().train()
    model.residual_scale.data.fill_(0.2)
    visual, global_semantic, aligned = _inputs()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(
            _packet(),
            visual_grid=visual,
            global_semantic=global_semantic,
            aligned_agent_tokens=aligned,
        )
        loss = output["joint_role_logits"].square().mean()
    assert output["joint_role_probs"].dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_no_person_scene_produces_no_visible_or_null_role_candidates():
    model = _model().eval()
    packet = PairPacket(
        pair_feats=torch.empty(1, 0, 8),
        dense_tokens=torch.randn(1, 8, 2, 2),
        subject_boxes=torch.empty(1, 0, 4),
        object_boxes=torch.empty(1, 0, 4),
        entity_boxes=torch.tensor([[[0.2, 0.2, 0.6, 0.6]]]),
        entity_feats=torch.randn(1, 1, 8),
        entity_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        subject_indices=torch.empty(1, 0, dtype=torch.long),
        object_indices=torch.empty(1, 0, dtype=torch.long),
        entity_labels=torch.tensor([[2]]),
    )
    output = model(
        packet,
        visual_grid=torch.randn(1, 6, 3, 3),
        global_semantic=torch.randn(1, 10),
        aligned_agent_tokens=torch.empty(1, 0, 6),
    )
    assert output["joint_role_probs"].shape == (1, 0, 3, 2)
    assert output["event_states"].shape == (1, 0, 3, 8)
    assert output["null_role_probs"].sum() == 0
