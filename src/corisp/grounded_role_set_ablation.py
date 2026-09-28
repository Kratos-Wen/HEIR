from __future__ import annotations

from dataclasses import dataclass

CORISP_V8_ABLATION_IDS = (
    "full",
    "no_line_graph",
    "no_cardinality",
    "hard_localization",
    "no_typed_null",
    "posthoc_detector",
)

@dataclass(frozen=True)
class GroundedRoleSetAblation:
    """Disable one grounded-set component at a time."""

    ablation_id: str = "full"
    use_line_graph: bool = True
    learn_cardinality: bool = True
    localization_target_mode: str = "marginal"
    use_typed_null_atoms: bool = True
    detector_score_mode: str = "inside"

    @classmethod
    def from_id(cls, ablation_id: str) -> "GroundedRoleSetAblation":
        controls = {
            "full": {},
            "no_line_graph": {"use_line_graph": False},
            "no_cardinality": {"learn_cardinality": False},
            "hard_localization": {"localization_target_mode": "best_iou"},
            "no_typed_null": {"use_typed_null_atoms": False},
            "posthoc_detector": {"detector_score_mode": "posthoc"},
        }
        try:
            changed = controls[str(ablation_id)]
        except KeyError as error:
            raise ValueError(
                f"Unknown ablation {ablation_id!r}; expected one of "
                f"{CORISP_V8_ABLATION_IDS}."
            ) from error
        return cls(ablation_id=str(ablation_id), **changed)

    def __post_init__(self) -> None:
        if self.ablation_id not in CORISP_V8_ABLATION_IDS:
            raise ValueError(f"Unknown ablation {self.ablation_id!r}.")
        if self.localization_target_mode not in ("marginal", "best_iou"):
            raise ValueError("Unsupported localization target mode.")
        if self.detector_score_mode not in ("inside", "posthoc"):
            raise ValueError("Unsupported detector score mode.")
