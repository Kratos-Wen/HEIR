import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corisp import (  # noqa: E402
    VCOCONullRoleIndex,
    VCOCONullRoleRecord,
    VCOCORoleSpace,
    load_vcoco_null_role_index,
)

sys.path.insert(0, str(ROOT / "integrations"))
from vcoco_null_dataset import (  # noqa: E402
    VCOCONullAwareDataFactory,
    include_official_vcoco_images,
)


def test_official_test_expansion_preserves_split_order_and_empty_images():
    class _Dataset:
        _image_ids = [11, 12, 13]
        _keep = [0, 2]
        _anno = [
            {"boxes_h": [[1, 2, 3, 4]], "boxes_o": [[2, 3, 4, 5]]},
            {"boxes_h": [], "boxes_o": []},
            {"boxes_h": [[3, 4, 5, 6]], "boxes_o": [[4, 5, 6, 7]]},
        ]

    class _Factory:
        name = "vcoco"
        dataset = _Dataset()

    include_official_vcoco_images(_Factory(), [13, 12, 11])
    assert _Factory.dataset._keep == [2, 1, 0]
    assert _Factory.dataset._anno[1]["boxes_h"].shape == (0, 4)
    assert _Factory.dataset._anno[1]["boxes_o"].shape == (0, 4)


def test_official_test_expansion_rejects_missing_images():
    class _Dataset:
        _image_ids = [11]
        _keep = [0]
        _anno = [{"boxes_h": [], "boxes_o": []}]

    class _Factory:
        name = "vcoco"
        dataset = _Dataset()

    try:
        include_official_vcoco_images(_Factory(), [11, 12])
    except ValueError as error:
        assert "missing 1 official images" in str(error)
    else:
        raise AssertionError("A partial official V-COCO split must be rejected.")


def test_null_role_loader_respects_column_major_vcoco_layout(tmp_path):
    role_space = VCOCORoleSpace.from_role_classes(
        ("hit instr", "hit obj", "hold obj")
    )
    vsrl = [
        {
            "action_name": "hit",
            "role_name": ["agent", "instr", "obj"],
            "ann_id": [10, 11],
            "label": [1, 1],
            # Columns are agent=[10,11], instr=[0,21], obj=[20,0].
            "role_object_id": [10, 11, 0, 21, 20, 0],
        }
    ]
    coco = {
        "annotations": [
            {"id": 10, "image_id": 7, "bbox": [1, 2, 3, 4]},
            {"id": 11, "image_id": 7, "bbox": [5, 6, 7, 8]},
        ]
    }
    vsrl_path = tmp_path / "vsrl.json"
    coco_path = tmp_path / "coco.json"
    vsrl_path.write_text(json.dumps(vsrl), encoding="utf-8")
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    index = load_vcoco_null_role_index(vsrl_path, coco_path, role_space)
    records = index.for_image(7)
    assert len(records) == 2
    assert records[0].agent_annotation_id == 10
    assert records[0].agent_box_xyxy == (1.0, 2.0, 4.0, 6.0)
    assert records[0].role_class_index == 0
    assert records[1].agent_annotation_id == 11
    assert records[1].role_class_index == 1


def test_null_only_image_preserves_integer_class_indices():
    class _Dataset:
        _image_ids = [7]
        _keep = []

        def __len__(self):
            return len(self._keep)

        def image_id(self, index):
            return self._image_ids[self._keep[index]]

        def __getitem__(self, index):
            return None, {
                "boxes_h": torch.empty(0, 4),
                "boxes_o": torch.empty(0, 4),
                "actions": torch.empty(0),
                "objects": torch.empty(0),
            }

    class _Factory:
        name = "vcoco"
        dataset = _Dataset()
        transforms = staticmethod(lambda image, target: (image, target))

    index = VCOCONullRoleIndex(
        [
            VCOCONullRoleRecord(
                image_id=7,
                agent_annotation_id=10,
                agent_box_xyxy=(1.0, 2.0, 4.0, 6.0),
                role_class_index=3,
                action_name="eat",
                role_name="obj",
            )
        ]
    )
    _, target = VCOCONullAwareDataFactory(_Factory(), index)[0]
    assert target["labels"].dtype == torch.long
    assert target["object"].dtype == torch.long
    assert target["labels"].numel() == 0
    assert target["null_role_classes"].dtype == torch.long
    assert target["null_role_classes"].tolist() == [3]


def test_source_agent_identity_survives_transform_box_collapse():
    class _Dataset:
        _image_ids = [7]
        _keep = [0]

        def __len__(self):
            return 1

        def image_id(self, index):
            return 7

        def __getitem__(self, index):
            return None, {
                "boxes_h": torch.tensor(
                    [[1.0, 2.0, 4.0, 6.0], [5.0, 6.0, 9.0, 10.0]]
                ),
                "boxes_o": torch.tensor(
                    [[2.0, 3.0, 5.0, 7.0], [6.0, 7.0, 10.0, 11.0]]
                ),
                "actions": torch.tensor([0, 0]),
                "objects": torch.tensor([3, 4]),
            }

    def collapse_people(image, target):
        target = target.copy()
        target["boxes_h"] = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0]]
        ).repeat(len(target["boxes_h"]), 1)
        return image, target

    class _Factory:
        name = "vcoco"
        dataset = _Dataset()
        transforms = staticmethod(collapse_people)

    index = VCOCONullRoleIndex(
        [
            VCOCONullRoleRecord(
                image_id=7,
                agent_annotation_id=10,
                agent_box_xyxy=(1.0, 2.0, 4.0, 6.0),
                role_class_index=1,
                action_name="hit",
                role_name="instr",
            )
        ]
    )
    _, target = VCOCONullAwareDataFactory(_Factory(), index)[0]
    assert target["object"].tolist() == [3, 4]
    assert target["agent_instance_ids"].tolist() == [0, 1]
    assert target["null_agent_instance_ids"].tolist() == [0]
