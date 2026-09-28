"""Recover V-COCO positive actions whose role filler is unannotated."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from corisp.vcoco_roles import VCOCORoleSpace


@dataclass(frozen=True)
class VCOCONullRoleRecord:
    image_id: int
    agent_annotation_id: int
    agent_box_xyxy: tuple[float, float, float, float]
    role_class_index: int
    action_name: str
    role_name: str


class VCOCONullRoleIndex:
    """Image-indexed immutable null-role supervision recovered from VSRL."""

    def __init__(self, records: Iterable[VCOCONullRoleRecord]) -> None:
        ordered = sorted(
            set(records),
            key=lambda item: (
                item.image_id,
                item.agent_annotation_id,
                item.role_class_index,
            ),
        )
        by_image: dict[int, list[VCOCONullRoleRecord]] = {}
        for record in ordered:
            by_image.setdefault(record.image_id, []).append(record)
        self._records = tuple(ordered)
        self._by_image = {
            image_id: tuple(values) for image_id, values in by_image.items()
        }

    def __len__(self) -> int:
        return len(self._records)

    @property
    def image_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_image))

    @property
    def records(self) -> tuple[VCOCONullRoleRecord, ...]:
        return self._records

    def for_image(self, image_id: int) -> tuple[VCOCONullRoleRecord, ...]:
        return self._by_image.get(int(image_id), ())

    def summary(self, role_space: VCOCORoleSpace) -> dict:
        counts = [0 for _ in range(role_space.num_role_classes)]
        for record in self._records:
            counts[record.role_class_index] += 1
        return {
            "schema": "corisp_vcoco_null_role_index_v1",
            "records": len(self._records),
            "images": len(self._by_image),
            "role_class_counts": dict(zip(role_space.role_class_names, counts)),
        }


def load_vcoco_null_role_index(
    vsrl_json: str | Path,
    coco_json: str | Path,
    role_space: VCOCORoleSpace,
) -> VCOCONullRoleIndex:
    """Load positive, role-specific null fillers from official V-COCO files.

    Official ``role_object_id`` arrays are serialized in column-major order:
    all agent IDs are followed by all IDs for the first semantic role, and so
    on.  Treating the flattened array as row-major silently corrupts labels.
    """

    vsrl = json.loads(Path(vsrl_json).read_text(encoding="utf-8"))
    coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    annotations = {int(item["id"]): item for item in coco["annotations"]}
    class_lookup = {
        (role_space.action_names[action_index], role_space.role_names[role_index]): class_index
        for class_index, (action_index, role_index) in enumerate(
            zip(role_space.role_class_to_action, role_space.role_class_to_role)
        )
    }

    records: list[VCOCONullRoleRecord] = []
    for action in vsrl:
        action_name = str(action["action_name"])
        role_names = [str(value) for value in action["role_name"]]
        labels = [int(value) for value in action["label"]]
        agent_annotation_ids = [int(value) for value in action["ann_id"]]
        count = len(labels)
        role_count = len(role_names)
        flat_role_ids = [int(value) for value in action["role_object_id"]]
        if len(flat_role_ids) != count * role_count:
            raise ValueError(
                f"Malformed role_object_id for V-COCO action {action_name!r}."
            )
        columns = [
            flat_role_ids[column * count : (column + 1) * count]
            for column in range(role_count)
        ]
        for row, (label, agent_annotation_id) in enumerate(
            zip(labels, agent_annotation_ids)
        ):
            if label != 1:
                continue
            serialized_agent = columns[0][row]
            if serialized_agent not in (0, agent_annotation_id):
                raise ValueError(
                    f"V-COCO agent ID mismatch for action {action_name!r}."
                )
            annotation = annotations.get(agent_annotation_id)
            if annotation is None:
                raise KeyError(
                    f"Missing COCO annotation {agent_annotation_id} for V-COCO."
                )
            x, y, width, height = (float(value) for value in annotation["bbox"])
            box = (x, y, x + width, y + height)
            for column, role_name in enumerate(role_names[1:], start=1):
                class_index = class_lookup.get((action_name, role_name))
                if class_index is None or columns[column][row] != 0:
                    continue
                records.append(
                    VCOCONullRoleRecord(
                        image_id=int(annotation["image_id"]),
                        agent_annotation_id=agent_annotation_id,
                        agent_box_xyxy=box,
                        role_class_index=class_index,
                        action_name=action_name,
                        role_name=role_name,
                    )
                )
    return VCOCONullRoleIndex(records)


__all__ = [
    "VCOCONullRoleIndex",
    "VCOCONullRoleRecord",
    "load_vcoco_null_role_index",
]
