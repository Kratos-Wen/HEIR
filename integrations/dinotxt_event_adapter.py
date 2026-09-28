from __future__ import annotations

import hashlib

from pathlib import Path

from typing import Sequence

import torch

import torch.nn as nn

import torch.nn.functional as F

DINO_VITL_BACKBONE_SHA256 = (
    "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
)

DINO_TXT_HEAD_SHA256 = (
    "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
)

def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_dinotxt_prototype_bank(
    path: str | Path,
    *,
    action_names: Sequence[str],
    object_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    supported_schemas = {
        "corisp_vcoco_dinotxt_prototypes_v1",
        "corisp_hico_dinotxt_prototypes_v1",
        "corisp_hico_dinotxt_hoi600_prototypes_v2",
    }
    if payload.get("schema") not in supported_schemas:
        raise ValueError("Unsupported DINO.txt prototype-bank schema.")
    if list(payload["action_names"]) != list(action_names):
        raise ValueError("DINO.txt action prototype order does not match the dataset.")
    if list(payload["object_names"]) != list(object_names):
        raise ValueError("DINO.txt object prototype order does not match the dataset.")
    action = payload["action_prototypes"].float()
    objects = payload["object_prototypes"].float()
    if action.shape != (len(action_names), 2048):
        raise ValueError("DINO.txt action prototypes must be [A,2048].")
    if objects.shape != (len(object_names), 2048):
        raise ValueError("DINO.txt object prototypes must be [O,2048].")
    return F.normalize(action, dim=-1), F.normalize(objects, dim=-1), payload

def load_dinotxt_role_prototype_bank(
    path: str | Path,
    *,
    role_names: Sequence[str],
) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "corisp_dinotxt_role_prototypes_v1":
        raise ValueError("Unsupported DINO.txt role-prototype schema.")
    if list(payload.get("role_names", [])) != list(role_names):
        raise ValueError("DINO.txt role prototype order does not match the ontology.")
    prototypes = payload["role_prototypes"].float()
    if prototypes.shape != (len(role_names), 2048):
        raise ValueError("DINO.txt role prototypes must be [R,2048].")
    if not torch.isfinite(prototypes).all():
        raise ValueError("DINO.txt role prototypes contain non-finite values.")
    return F.normalize(prototypes, dim=-1), payload

class FrozenDINOtxtVisualBackbone(nn.Module):
    """Load only the official frozen DINOv3 ViT-L and DINO.txt vision head.

    The 1.28B text tower is intentionally not constructed in every DDP worker;
    text prototypes are generated once and loaded from a hashed artifact.
    """

    output_dim = 1024
    semantic_dim = 2048

    def __init__(
        self,
        *,
        backbone_weights: str | Path,
        dinotxt_weights: str | Path,
        input_size: int = 448,
        verify_hashes: bool = True,
        storage_dtype: torch.dtype = torch.bfloat16,
        head_compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        backbone_weights = Path(backbone_weights)
        dinotxt_weights = Path(dinotxt_weights)
        if not backbone_weights.is_file() or not dinotxt_weights.is_file():
            raise FileNotFoundError("Pinned DINOv3/DINO.txt weights are missing.")
        if input_size <= 0 or input_size % 16:
            raise ValueError("DINO input_size must be a positive multiple of 16.")
        if verify_hashes:
            actual_backbone = _sha256(backbone_weights)
            actual_head = _sha256(dinotxt_weights)
            if actual_backbone != DINO_VITL_BACKBONE_SHA256:
                raise ValueError(
                    "DINOv3 ViT-L checkpoint hash mismatch: "
                    f"{actual_backbone}."
                )
            if actual_head != DINO_TXT_HEAD_SHA256:
                raise ValueError(
                    "DINO.txt checkpoint hash mismatch: " f"{actual_head}."
                )

        from models.dinov3.eval.text.vision_tower import VisionHead
        from models.dinov3.hub.backbones import dinov3_vitl16

        backbone = dinov3_vitl16(pretrained=False)
        backbone_state = torch.load(
            backbone_weights,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        backbone.load_state_dict(backbone_state, strict=True)
        head = VisionHead(
            input_dim=int(backbone.embed_dim),
            embed_dim=self.semantic_dim,
            num_heads=int(backbone.num_heads),
            num_blocks=2,
            blocks_drop_path=0.3,
            use_class_token=True,
            use_patch_tokens=True,
            use_linear_projection=False,
        )
        complete_state = torch.load(
            dinotxt_weights,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        prefix = "visual_model.head."
        head_state = {
            key[len(prefix) :]: value
            for key, value in complete_state.items()
            if key.startswith(prefix)
        }
        head.load_state_dict(head_state, strict=True)
        if head_compute_dtype != torch.float32:
            raise ValueError(
                "The frozen DINO.txt semantic bridge must compute in FP32."
            )
        self.backbone = backbone.to(dtype=storage_dtype)
        # Event-query gradients cross this frozen head even though its weights
        # are not updated. Keeping the two alignment blocks in FP32 prevents
        # BF16 attention-backward overflow without changing the frozen model.
        self.head = head.to(dtype=head_compute_dtype)
        self.num_register_tokens = int(
            getattr(backbone, "n_storage_tokens", getattr(backbone, "num_register_tokens", 0))
        )
        self.input_size = int(input_size)
        self.storage_dtype = storage_dtype
        self.head_compute_dtype = head_compute_dtype
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenDINOtxtVisualBackbone":
        super().train(False)
        return self

    def _forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the frozen semantic bridge in FP32 under any outer autocast."""

        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return self.head(tokens.to(dtype=self.head_compute_dtype))

    @torch.no_grad()
    def _backbone_tokens(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Each normalized DINO image must be [3,H,W].")
        pixels = F.interpolate(
            image[None],
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(dtype=self.storage_dtype)
        intermediate = self.backbone.get_intermediate_layers(
            pixels,
            n=1,
            return_class_token=True,
            return_extra_tokens=True,
        )[-1]
        patch_tokens, class_token, register_tokens = intermediate
        return torch.cat(
            [class_token[:, None], register_tokens, patch_tokens], dim=1
        ).detach()

    @staticmethod
    def _aligned_outputs(
        aligned: torch.Tensor,
        *,
        patch_start: int,
        input_size: int,
        output_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_class = aligned[:, 0]
        aligned_patch = aligned[:, patch_start:]
        side = input_size // 16
        if aligned_patch.shape[1] != side * side:
            raise RuntimeError("DINOv3 patch count does not match the pinned input grid.")
        grid = aligned_patch.reshape(1, side, side, output_dim)
        grid = F.normalize(grid.float(), dim=-1).permute(0, 3, 1, 2)
        semantic = torch.cat(
            [aligned_class.float(), aligned_patch.float().mean(dim=1)], dim=-1
        )
        return grid, F.normalize(semantic, dim=-1)

    @torch.no_grad()
    def forward(
        self, images: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grids: list[torch.Tensor] = []
        semantics: list[torch.Tensor] = []
        for image in images:
            tokens = self._backbone_tokens(image)
            aligned = self._forward_head(tokens)
            grid, semantic = self._aligned_outputs(
                aligned,
                patch_start=self.num_register_tokens + 1,
                input_size=self.input_size,
                output_dim=self.output_dim,
            )
            grids.append(grid)
            semantics.append(semantic)
        return torch.cat(grids, dim=0), torch.cat(semantics, dim=0)

    def forward_with_event_queries(
        self,
        images: Sequence[torch.Tensor],
        event_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jointly align trainable event queries and frozen image tokens.

        The DINOv3 backbone remains under ``no_grad``.  The frozen DINO.txt
        head is evaluated with autograd enabled so gradients flow through it
        into the event-query projection, matching the semantic bridge used by
        SL-HOI without updating any DINO parameter.
        """

        if event_queries.ndim != 3 or event_queries.shape[0] != len(images):
            raise ValueError("event_queries must be [B,G,1024] and match images.")
        if event_queries.shape[-1] != self.output_dim:
            raise ValueError(
                f"event query width must be {self.output_dim}, got {event_queries.shape[-1]}."
            )
        aligned_events: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        semantics: list[torch.Tensor] = []
        for index, image in enumerate(images):
            tokens = self._backbone_tokens(image).to(dtype=self.head_compute_dtype)
            query = event_queries[index : index + 1].to(
                dtype=self.head_compute_dtype
            )
            query_start = self.num_register_tokens + 1
            query_end = query_start + query.shape[1]
            joint_tokens = torch.cat(
                [tokens[:, :query_start], query, tokens[:, query_start:]], dim=1
            )
            aligned = self._forward_head(joint_tokens)
            event = aligned[:, query_start:query_end].float()
            grid, semantic = self._aligned_outputs(
                aligned,
                patch_start=query_end,
                input_size=self.input_size,
                output_dim=self.output_dim,
            )
            aligned_events.append(event)
            grids.append(grid)
            semantics.append(semantic)
        return (
            torch.cat(aligned_events, dim=0),
            torch.cat(grids, dim=0),
            torch.cat(semantics, dim=0),
        )

    def forward_with_reference_and_binding_queries(
        self,
        images: Sequence[torch.Tensor],
        reference_queries: torch.Tensor,
        binding_queries: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Read one frozen backbone with agent and binding query sequences.

        The reference sequence contains only agent queries. The binding
        sequence additionally contains grounded pair and predicate-role frame
        queries.  Both passes share the same detached DINOv3 tokens and frozen
        DINO.txt head; gradients flow only into their query projections.
        """

        for name, query in (
            ("reference_queries", reference_queries),
            ("binding_queries", binding_queries),
        ):
            if query.ndim != 3 or query.shape[0] != len(images):
                raise ValueError(f"{name} must be [B,G,1024] and match images.")
            if query.shape[-1] != self.output_dim:
                raise ValueError(
                    f"{name} width must be {self.output_dim}, got {query.shape[-1]}."
                )

        reference_events: list[torch.Tensor] = []
        reference_grids: list[torch.Tensor] = []
        reference_semantics: list[torch.Tensor] = []
        binding_events: list[torch.Tensor] = []
        binding_grids: list[torch.Tensor] = []
        binding_semantics: list[torch.Tensor] = []
        query_start = self.num_register_tokens + 1
        for index, image in enumerate(images):
            tokens = self._backbone_tokens(image).to(dtype=self.head_compute_dtype)
            for query_bank, events, grids, semantics in (
                (
                    reference_queries,
                    reference_events,
                    reference_grids,
                    reference_semantics,
                ),
                (
                    binding_queries,
                    binding_events,
                    binding_grids,
                    binding_semantics,
                ),
            ):
                query = query_bank[index : index + 1].to(
                    dtype=self.head_compute_dtype
                )
                query_end = query_start + query.shape[1]
                joint_tokens = torch.cat(
                    [tokens[:, :query_start], query, tokens[:, query_start:]],
                    dim=1,
                )
                aligned = self._forward_head(joint_tokens)
                event = aligned[:, query_start:query_end].float()
                grid, semantic = self._aligned_outputs(
                    aligned,
                    patch_start=query_end,
                    input_size=self.input_size,
                    output_dim=self.output_dim,
                )
                events.append(event)
                grids.append(grid)
                semantics.append(semantic)
        return (
            torch.cat(reference_events, dim=0),
            torch.cat(reference_grids, dim=0),
            torch.cat(reference_semantics, dim=0),
            torch.cat(binding_events, dim=0),
            torch.cat(binding_grids, dim=0),
            torch.cat(binding_semantics, dim=0),
        )
