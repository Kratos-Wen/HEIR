import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import VCOCORoleSpace  # noqa: E402
from corisp.joint import CoRISPJointConfig, PriorResidualJointField  # noqa: E402


ROLE_CLASSES = (
    "hold obj",
    "sit instr",
    "ride instr",
    "look obj",
    "hit instr",
    "hit obj",
    "eat obj",
    "eat instr",
    "jump instr",
    "lay instr",
    "talk_on_phone instr",
    "carry obj",
    "throw obj",
    "catch obj",
    "cut instr",
    "cut obj",
    "work_on_computer instr",
    "ski instr",
    "surf instr",
    "skateboard instr",
    "drink instr",
    "kick obj",
    "read obj",
    "snowboard instr",
)


def test_vcoco_role_space_is_lossless_and_masks_illegal_slots():
    counts = tuple(range(1, len(ROLE_CLASSES) + 1))
    space = VCOCORoleSpace.from_role_classes(ROLE_CLASSES, counts)
    assert space.num_role_classes == 24
    assert space.num_actions == 21
    assert space.role_names == ("obj", "instr")
    valid = space.valid_role_mask()
    assert valid.sum().item() == 24
    for action in ("hit", "eat", "cut"):
        assert valid[space.action_names.index(action)].tolist() == [True, True]
    assert valid[space.action_names.index("hold")].tolist() == [True, False]
    assert valid[space.action_names.index("sit")].tolist() == [False, True]

    prior = space.semantic_role_prior()
    assert torch.allclose(prior.sum(dim=-1), torch.ones(space.num_actions))
    assert torch.equal(prior.eq(0), ~valid)

    object_mask = space.observed_object_action_mask(
        [[ROLE_CLASSES.index("hold obj")], [ROLE_CLASSES.index("sit instr")]]
    )
    assert object_mask.shape == (2, space.num_actions)
    assert object_mask[0, space.action_names.index("hold")]
    assert not object_mask[0, space.action_names.index("sit")]
    assert object_mask[1, space.action_names.index("sit")]


def test_vcoco_labels_collapse_and_scores_expand_in_official_order():
    space = VCOCORoleSpace.from_role_classes(ROLE_CLASSES)
    labels = torch.zeros(2, space.num_role_classes)
    labels[0, ROLE_CLASSES.index("hit instr")] = 1
    labels[0, ROLE_CLASSES.index("hit obj")] = 1
    labels[1, ROLE_CLASSES.index("hold obj")] = 1
    action_labels, acceptable = space.collapse_pair_labels(labels)
    hit = space.action_names.index("hit")
    hold = space.action_names.index("hold")
    assert action_labels[0, hit] == 1
    assert acceptable[0, hit].tolist() == [True, True]
    assert action_labels[1, hold] == 1
    assert acceptable[1, hold].tolist() == [True, False]

    action_probability = torch.full((2, space.num_actions), 0.8)
    conditional = space.valid_role_mask().float()
    conditional = conditional / conditional.sum(dim=-1, keepdim=True)
    conditional = conditional.expand(2, -1, -1)
    expanded = space.expand_role_scores(action_probability, conditional)
    assert expanded.shape == (2, 24)
    assert torch.allclose(
        expanded[:, ROLE_CLASSES.index("hold obj")], torch.full((2,), 0.8)
    )
    assert torch.allclose(
        expanded[:, ROLE_CLASSES.index("hit obj")], torch.full((2,), 0.4)
    )
    assert torch.allclose(
        expanded[:, ROLE_CLASSES.index("hit instr")], torch.full((2,), 0.4)
    )


def test_joint_field_preserves_exact_marginal_with_invalid_role_states():
    valid = torch.tensor([[True, False], [True, True]])
    prior = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    field = PriorResidualJointField(
        CoRISPJointConfig(
            d_model=8,
            num_hoi_classes=2,
            roles=("obj", "instr"),
            num_bases=2,
            hidden_dim=16,
            evidence_dim=8,
            semantic_prior=prior,
            semantic_role_valid_mask=valid,
        )
    )
    base = torch.tensor([[[0.2, -0.4]]])
    zeros = torch.zeros_like(base)
    residual = torch.zeros(1, 1, 2, 2)
    log_prior = field._log_prior(base.dtype, base.device)
    state = field._state_outputs(base, zeros, residual, log_prior)
    assert torch.allclose(state["hoi_logits"], base)
    assert state["conditional_role_probs"][0, 0, 0, 1] == 0

    finalized = field.finalize(
        base,
        zeros,
        {
            "joint_residual_logits": residual,
            "complement_joint_residual_logits": residual,
        },
    )
    neutral_prob = finalized["neutral_conditional_role_logits"].softmax(dim=-1)
    assert neutral_prob[0, 0, 0, 1] == 0
    neutral_states = finalized["neutral_joint_state_logits"]
    neutral_hoi = torch.logsumexp(neutral_states[..., 1:], dim=-1) - neutral_states[..., 0]
    assert torch.allclose(neutral_hoi, base)
