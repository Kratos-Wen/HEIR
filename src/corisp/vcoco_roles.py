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

    def semantic_role_prior(self, smoothing: float = 1.0) -> torch.Tensor:
        if smoothing <= 0:
            raise ValueError("V-COCO semantic-prior smoothing must be positive.")
        prior = torch.zeros(self.num_actions, self.num_roles, dtype=torch.float32)
        for class_idx, (action_idx, role_idx) in enumerate(
            zip(self.role_class_to_action, self.role_class_to_role)
        ):
            prior[action_idx, role_idx] = float(self.role_class_counts[class_idx]) + smoothing
        valid = self.valid_role_mask()
        prior = prior.masked_fill(~valid, 0.0)
        return prior / prior.sum(dim=-1, keepdim=True)

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

    def object_action_mask(
        self,
        object_to_role_classes: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Backward-compatible alias for :meth:`observed_object_action_mask`."""

        return self.observed_object_action_mask(object_to_role_classes)

    def collapse_pair_labels(
        self,
        role_class_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if role_class_labels.ndim != 2 or role_class_labels.shape[1] != self.num_role_classes:
            raise ValueError(
                "role_class_labels must be [N, C_role] in the official V-COCO order."
            )
        action_labels = role_class_labels.new_zeros(
            role_class_labels.shape[0], self.num_actions
        )
        acceptable_roles = role_class_labels.new_zeros(
            role_class_labels.shape[0], self.num_actions, self.num_roles
        )
        for class_idx, (action_idx, role_idx) in enumerate(
            zip(self.role_class_to_action, self.role_class_to_role)
        ):
            values = role_class_labels[:, class_idx]
            action_labels[:, action_idx] = torch.maximum(
                action_labels[:, action_idx], values
            )
            acceptable_roles[:, action_idx, role_idx] = torch.maximum(
                acceptable_roles[:, action_idx, role_idx], values
            )
        return action_labels, acceptable_roles

    def expand_role_scores(
        self,
        action_probabilities: torch.Tensor,
        conditional_role_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        expected = (*action_probabilities.shape, self.num_roles)
        if tuple(conditional_role_probabilities.shape) != expected:
            raise ValueError(
                "conditional_role_probabilities must align with action probabilities; "
                f"expected {expected}, got {tuple(conditional_role_probabilities.shape)}."
            )
        if action_probabilities.shape[-1] != self.num_actions:
            raise ValueError("Action probability count does not match V-COCO role space.")
        joint = action_probabilities[..., None] * conditional_role_probabilities
        return torch.stack(
            [
                joint[..., action_idx, role_idx]
                for action_idx, role_idx in zip(
                    self.role_class_to_action, self.role_class_to_role
                )
            ],
            dim=-1,
        )

    def as_dict(self) -> dict:
        return {
            "schema": "corisp_vcoco_role_space_v1",
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
            "semantic_role_prior_laplace_1": self.semantic_role_prior(1.0).tolist(),
        }
