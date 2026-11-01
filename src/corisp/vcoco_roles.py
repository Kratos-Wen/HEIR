from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import torch

VCOCO_ROLE_NAMES = ("obj", "instr")

@dataclass(frozen=True)
class VCOCORoleSpace:
    role_class_names: tuple[str, ...]
    action_names: tuple[str, ...]
    role_names: tuple[str, ...]
    role_class_to_action: tuple[int, ...]
    role_class_to_role: tuple[int, ...]
    action_role_to_role_class: tuple[tuple[int, ...], ...]
    role_class_counts: tuple[int, ...]

    @classmethod
    def from_role_classes(
        cls,
        role_class_names: Sequence[str],
        role_class_counts: Sequence[int] | None = None,
        *,
        role_names: Sequence[str] = VCOCO_ROLE_NAMES,
    ) -> "VCOCORoleSpace":
        classes = tuple(str(name).strip() for name in role_class_names)
        roles = tuple(str(name).strip() for name in role_names)
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("V-COCO role classes must be non-empty and unique.")
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("V-COCO role names must be non-empty and unique.")

        action_names: list[str] = []
        parsed: list[tuple[str, str]] = []
        for name in classes:
            try:
                action, role = name.rsplit(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid V-COCO role class {name!r}.") from exc
            if not action or role not in roles:
                raise ValueError(
                    f"Invalid V-COCO role class {name!r}; expected '<action> {roles}'."
                )
            parsed.append((action, role))
            if action not in action_names:
                action_names.append(action)

        action_index = {name: index for index, name in enumerate(action_names)}
        role_index = {name: index for index, name in enumerate(roles)}
        class_to_action = tuple(action_index[action] for action, _ in parsed)
        class_to_role = tuple(role_index[role] for _, role in parsed)
        reverse = [[-1 for _ in roles] for _ in action_names]
        for class_index, (action_idx, role_idx) in enumerate(
            zip(class_to_action, class_to_role)
        ):
            if reverse[action_idx][role_idx] != -1:
                raise ValueError(
                    "V-COCO contains duplicate action-role class "
                    f"{classes[class_index]!r}."
                )
            reverse[action_idx][role_idx] = class_index

        if role_class_counts is None:
            counts = tuple(1 for _ in classes)
        else:
            counts = tuple(int(value) for value in role_class_counts)
            if len(counts) != len(classes) or any(value < 0 for value in counts):
                raise ValueError("role_class_counts must be non-negative and match class order.")

        return cls(
            role_class_names=classes,
            action_names=tuple(action_names),
            role_names=roles,
            role_class_to_action=class_to_action,
            role_class_to_role=class_to_role,
            action_role_to_role_class=tuple(tuple(row) for row in reverse),
            role_class_counts=counts,
        )

    @property
    def num_role_classes(self) -> int:
        return len(self.role_class_names)

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    @property
    def num_roles(self) -> int:
        return len(self.role_names)

    def valid_role_mask(self) -> torch.Tensor:
        return torch.tensor(self.action_role_to_role_class, dtype=torch.long).ge(0)

    def collapse_object_to_actions(
        self,
        object_to_role_classes: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        collapsed: list[list[int]] = []
        for role_classes in object_to_role_classes:
            actions = {
                self.role_class_to_action[int(class_idx)] for class_idx in role_classes
            }
            collapsed.append(sorted(actions))
        return collapsed

    def observed_object_action_mask(
        self,
        object_to_role_classes: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Return object-action support observed in the loaded V-COCO split.

        ``VCOCO.object_to_action`` is derived from positive annotations in the
        loaded partition.  It is the compatibility prior used by the matched
        PViC/H-DETR protocol, not a semantic ontology and not evidence that an
        unobserved composition is impossible.
        """

        collapsed = self.collapse_object_to_actions(object_to_role_classes)
        mask = torch.zeros(len(collapsed), self.num_actions, dtype=torch.bool)
        for object_index, actions in enumerate(collapsed):
            mask[object_index, actions] = True
        return mask

    def as_dict(self) -> dict:
        return {
            "role_class_names": list(self.role_class_names),
            "action_names": list(self.action_names),
            "role_names": list(self.role_names),
            "role_class_to_action": list(self.role_class_to_action),
            "role_class_to_role": list(self.role_class_to_role),
            "action_role_to_role_class": [
                list(row) for row in self.action_role_to_role_class
            ],
            "role_class_counts": list(self.role_class_counts),
            "valid_role_mask": self.valid_role_mask().tolist(),
        }
