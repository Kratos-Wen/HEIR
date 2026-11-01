import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import VCOCORoleSpace  # noqa: E402


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

    object_mask = space.observed_object_action_mask(
        [[ROLE_CLASSES.index("hold obj")], [ROLE_CLASSES.index("sit instr")]]
    )
    assert object_mask.shape == (2, space.num_actions)
    assert object_mask[0, space.action_names.index("hold")]
    assert not object_mask[0, space.action_names.index("sit")]
    assert object_mask[1, space.action_names.index("sit")]
