from __future__ import annotations

from dataclasses import replace

from pathlib import Path

from typing import List, Optional, Sequence

import torch

import torch.distributed as dist

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor

from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list

from ops import associate_with_ground_truth, binary_focal_loss_with_logits, compute_prior_scores, prepare_region_proposals

from corisp import PreparedPairTargetConfig, PreparedProposalPairAdapter, CoRISPJointLossConfig, CoRISPStrongDecoder, build_prepared_pair_joint_targets, corisp_joint_loss

IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)

CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def _inverse_sigmoid(value: Tensor, eps: float = 1e-5) -> Tensor:
    value = value.clamp(min=0.0, max=1.0)
    return torch.log(value.clamp(min=eps) / (1.0 - value).clamp(min=eps))

def _normalize_proposal_precision(
    proposals: list[dict[str, Tensor]],
) -> list[dict[str, Tensor]]:
    """Keep proposal geometry/confidence compatible with FP32 prior operators."""

    return [
        {
            **proposal,
            "boxes": proposal["boxes"].float(),
            "scores": proposal["scores"].float(),
        }
        for proposal in proposals
    ]

def _hdetr_forward_with_tokens(
    detector: nn.Module,
    samples: List[Tensor] | Tensor | NestedTensor,
) -> tuple[dict[str, Tensor], Tensor, list[NestedTensor]]:
    """Run the official H-DETR path while retaining proposal/backbone tokens."""

    if not isinstance(samples, NestedTensor):
        samples = nested_tensor_from_tensor_list(samples)
    features, positional = detector.backbone(samples)

    sources: list[Tensor] = []
    masks: list[Tensor] = []
    for level, feature in enumerate(features):
        source, mask = feature.decompose()
        if mask is None:
            raise RuntimeError("H-DETR returned a feature level without a padding mask.")
        sources.append(detector.input_proj[level](source))
        masks.append(mask)
    if detector.num_feature_levels > len(sources):
        initial_levels = len(sources)
        for level in range(initial_levels, detector.num_feature_levels):
            if level == initial_levels:
                source = detector.input_proj[level](features[-1].tensors)
            else:
                source = detector.input_proj[level](sources[-1])
            mask = F.interpolate(
                samples.mask[None].float(), size=source.shape[-2:]
            ).to(torch.bool)[0]
            pos_level = detector.backbone[1](NestedTensor(source, mask)).to(source.dtype)
            sources.append(source)
            masks.append(mask)
            positional.append(pos_level)

    query_embeddings = None
    if not detector.two_stage or detector.mixed_selection:
        query_embeddings = detector.query_embed.weight[: detector.num_queries]

    attention_mask = torch.zeros(
        detector.num_queries,
        detector.num_queries,
        dtype=torch.bool,
        device=sources[0].device,
    )
    split = detector.num_queries_one2one
    attention_mask[split:, :split] = True
    attention_mask[:split, split:] = True

    (
        hidden_states,
        initial_reference,
        intermediate_references,
        encoder_class,
        encoder_boxes_unactivated,
    ) = detector.transformer(
        sources,
        masks,
        positional,
        query_embeddings,
        attention_mask,
    )

    classes_one2one: list[Tensor] = []
    boxes_one2one: list[Tensor] = []
    classes_one2many: list[Tensor] = []
    boxes_one2many: list[Tensor] = []
    for level in range(hidden_states.shape[0]):
        reference = initial_reference if level == 0 else intermediate_references[level - 1]
        reference = _inverse_sigmoid(reference)
        class_logits = detector.class_embed[level](hidden_states[level])
        box_delta = detector.bbox_embed[level](hidden_states[level])
        if reference.shape[-1] == 4:
            box_delta = box_delta + reference
        else:
            box_delta = box_delta.clone()
            box_delta[..., :2] = box_delta[..., :2] + reference
        boxes = box_delta.sigmoid()
        classes_one2one.append(class_logits[:, :split])
        boxes_one2one.append(boxes[:, :split])
        classes_one2many.append(class_logits[:, split:])
        boxes_one2many.append(boxes[:, split:])

    output = {
        "pred_logits": classes_one2one[-1],
        "pred_boxes": boxes_one2one[-1],
        "pred_logits_one2many": classes_one2many[-1],
        "pred_boxes_one2many": boxes_one2many[-1],
    }
    if detector.two_stage:
        output["enc_outputs"] = {
            "pred_logits": encoder_class,
            "pred_boxes": encoder_boxes_unactivated.sigmoid(),
        }
    return output, hidden_states, features

class FrozenCLIPVisualBackbone(nn.Module):
    """Official pretrained CLIP vision tower with token-preserving output."""

    def __init__(self, model_name_or_path: str | Path) -> None:
        super().__init__()
        import clip

        model, _ = clip.load(str(model_name_or_path), device="cpu", jit=False)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.visual = model.visual
        self.image_size = int(self.visual.input_resolution)
        projection = getattr(self.visual, "proj", None)
        if projection is None or projection.ndim != 2:
            raise RuntimeError("The CLIP visual tower does not expose its output projection.")
        self.output_dim = int(projection.shape[-1])
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "FrozenCLIPVisualBackbone":
        super().train(False)
        self.visual.eval()
        return self

    def _prepare_images(self, images: Sequence[Tensor]) -> Tensor:
        prepared: list[Tensor] = []
        mean = self.imagenet_mean[0]
        std = self.imagenet_std[0]
        for image in images:
            rgb = (image * std + mean).clamp(0.0, 1.0)
            rgb = F.interpolate(
                rgb[None],
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            prepared.append(rgb)
        pixels = torch.stack(prepared, dim=0)
        return (pixels - self.clip_mean) / self.clip_std

    @torch.no_grad()
    def forward(self, images: Sequence[Tensor]) -> Tensor:
        pixels = self._prepare_images(images)
        dtype = next(self.visual.parameters()).dtype
        tokens = self.visual(pixels.to(dtype=dtype))
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        if tokens.ndim != 3 or tokens.shape[-1] != self.output_dim:
            raise RuntimeError(
                "CLIP must return token-preserving [B,T,D] features; "
                f"received {tuple(tokens.shape)}."
            )
        return F.normalize(tokens.float(), dim=-1)

class CLIPVisualPairAdapter(nn.Module):
    """Fuse frozen generic CLIP visual tokens without MGNM/UniHOI modules."""

    def __init__(self, clip_dim: int, d_model: int, residual_init: float = 0.1) -> None:
        super().__init__()
        self.clip_dim = int(clip_dim)
        self.d_model = int(d_model)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(self.clip_dim),
            nn.Linear(self.clip_dim, self.d_model),
        )
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(5 * self.d_model),
            nn.Linear(5 * self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Linear(2 * self.d_model, self.d_model),
        )
        self.pair_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.endpoint_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.dense_scale = nn.Parameter(torch.tensor(float(residual_init)))

    @staticmethod
    def _patch_grid(tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError("CLIP tokens must contain one class token and patch tokens.")
        count = int(tokens.shape[1] - 1)
        side = int(round(count**0.5))
        if side * side != count:
            raise ValueError(f"CLIP patch token count {count} is not a square grid.")
        return tokens[:, 1:].reshape(tokens.shape[0], side, side, tokens.shape[-1])

    @staticmethod
    def _pool_boxes(patch_grid: Tensor, boxes: Tensor) -> Tensor:
        if patch_grid.ndim != 4 or boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("patch_grid and boxes must be [B,H,W,D] and [B,N,4].")
        b, height, width, dim = patch_grid.shape
        if boxes.shape[0] != b:
            raise ValueError("CLIP patch grid and boxes must share the batch dimension.")
        y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
        x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xx = xx[None, None]
        yy = yy[None, None]
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        mask = (
            (xx >= x1[..., None, None])
            & (xx <= x2[..., None, None])
            & (yy >= y1[..., None, None])
            & (yy <= y2[..., None, None])
        )
        weights = mask.to(dtype=patch_grid.dtype)
        counts = weights.sum(dim=(-2, -1), keepdim=True)
        pooled = torch.einsum(
            "bnhw,bhwd->bnd",
            weights / counts.clamp_min(1.0),
            patch_grid,
        )
        empty = counts.flatten(2).squeeze(-1) == 0
        if empty.any():
            centers_x = ((x1 + x2) * 0.5 * width).floor().long().clamp(0, width - 1)
            centers_y = ((y1 + y2) * 0.5 * height).floor().long().clamp(0, height - 1)
            batch = torch.arange(b, device=boxes.device)[:, None].expand_as(centers_x)
            nearest = patch_grid[batch, centers_y, centers_x]
            pooled = torch.where(empty[..., None], nearest, pooled)
        return pooled.reshape(b, boxes.shape[1], dim)

    def forward(self, packet, clip_tokens: Tensor):
        if clip_tokens.ndim != 3 or clip_tokens.shape[-1] != self.clip_dim:
            raise ValueError(f"clip_tokens must be [B,T,{self.clip_dim}].")
        projected = self.token_projection(clip_tokens.to(dtype=packet.pair_feats.dtype))
        patch_grid = self._patch_grid(projected)
        subject = self._pool_boxes(patch_grid, packet.subject_boxes)
        obj = self._pool_boxes(patch_grid, packet.object_boxes)
        global_context = projected[:, :1].expand(-1, packet.num_pairs, -1)
        pair_delta = self.pair_projection(
            torch.cat(
                [packet.pair_feats, subject, obj, subject * obj, global_context],
                dim=-1,
            )
        )
        pair_feats = packet.pair_feats + self.pair_scale.to(packet.pair_feats.dtype) * pair_delta

        subject_feats = packet.subject_feats
        if subject_feats is not None:
            subject_feats = subject_feats + self.endpoint_scale.to(subject.dtype) * subject
        object_feats = packet.object_feats
        if object_feats is not None:
            object_feats = object_feats + self.endpoint_scale.to(obj.dtype) * obj

        entity_feats = packet.entity_feats
        if entity_feats is not None and packet.entity_boxes is not None:
            entity_context = self._pool_boxes(patch_grid, packet.entity_boxes)
            entity_feats = entity_feats + self.endpoint_scale.to(entity_context.dtype) * entity_context

        clip_dense = patch_grid.permute(0, 3, 1, 2)
        clip_dense = F.interpolate(
            clip_dense,
            size=packet.dense_tokens.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dense_tokens = packet.dense_tokens + self.dense_scale.to(clip_dense.dtype) * clip_dense
        return replace(
            packet,
            pair_feats=pair_feats,
            dense_tokens=dense_tokens,
            subject_feats=subject_feats,
            object_feats=object_feats,
            entity_feats=entity_feats,
            source="frozen_h_detr_swin_l_plus_frozen_clip_l14_336_corisp_only",
        )

class HDETRCoRISPv3(nn.Module):
    """Minimal H-DETR pair host with an optional CoRISP-only interaction head."""

    def __init__(
        self,
        detector: nn.Module,
        postprocessor: nn.Module,
        adapter: PreparedProposalPairAdapter,
        classifier: nn.Linear,
        object_to_target: Sequence[Sequence[int]],
        *,
        strong: CoRISPStrongDecoder | None,
        variant: str,
        clip_backbone: FrozenCLIPVisualBackbone | None = None,
        clip_adapter: CLIPVisualPairAdapter | None = None,
        human_idx: int = 0,
        box_score_thresh: float = 0.05,
        min_instances: int = 3,
        max_instances: int = 15,
        raw_lambda: float = 1.7,
        alpha: float = 0.5,
        gamma: float = 0.1,
        participation_loss_weight: float = 0.25,
        supervision_mode: str = "hico_only",
        joint_loss_cfg: CoRISPJointLossConfig | None = None,
        prepared_target_cfg: PreparedPairTargetConfig | None = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.adapter = adapter
        self.binary_classifier = classifier
        self.object_to_target = object_to_target
        self.strong = strong
        self._variant = str(variant)
        self.clip_backbone = clip_backbone
        self.clip_adapter = clip_adapter
        self.human_idx = int(human_idx)
        self.num_verbs = int(classifier.out_features)
        self.num_classes = self.num_verbs
        self.box_score_thresh = float(box_score_thresh)
        self.min_instances = int(min_instances)
        self.max_instances = int(max_instances)
        self.raw_lambda = float(raw_lambda)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.participation_loss_weight = float(participation_loss_weight)
        self.supervision_mode = str(supervision_mode)
        if self.supervision_mode not in {"hico_only", "heir_full", "vcoco_roles"}:
            raise ValueError(
                "supervision_mode must be 'hico_only', 'heir_full', or 'vcoco_roles'."
            )
        if self.supervision_mode == "heir_full" and self.strong is None:
            raise ValueError("This supervision mode requires a joint decoder.")
        self.joint_loss_cfg = joint_loss_cfg or CoRISPJointLossConfig()
        self.prepared_target_cfg = prepared_target_cfg or PreparedPairTargetConfig()
        self._last_corisp_metrics: dict[str, Tensor] = {}

        if self.strong is not None and self.supervision_mode == "hico_only":
            if classifier.in_features != self.strong.cfg.joint.d_model:
                raise ValueError("The CoRISP dimension must match the semantic classifier input.")
            residual_scale = self.strong.joint_decoder.joint_field.residual_scale
            residual_scale.data.zero_()
            residual_scale.requires_grad_(False)
        elif self.strong is not None and classifier.in_features != self.strong.cfg.joint.d_model:
            raise ValueError("The CoRISP dimension must match the semantic classifier input.")

    @property
    def variant(self) -> str:
        return self._variant

    def freeze_detector(self) -> None:
        self.detector.eval()
        for parameter in self.detector.parameters():
            parameter.requires_grad_(False)
        if self.clip_backbone is not None:
            self.clip_backbone.eval()
            for parameter in self.clip_backbone.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "HDETRCoRISPv3":
        super().train(mode)
        self.detector.eval()
        if self.clip_backbone is not None:
            self.clip_backbone.eval()
        return self

    def _extract_packets(
        self,
        images: List[Tensor],
    ) -> tuple[
        Tensor,
        list[dict[str, Tensor]],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list,
    ]:
        image_sizes = torch.as_tensor(
            [image.shape[-2:] for image in images], device=images[0].device
        )
        with torch.no_grad():
            detector_output, hidden_states, features = _hdetr_forward_with_tokens(
                self.detector, images
            )
            detections = self.postprocessor(detector_output, image_sizes)
            proposals = prepare_region_proposals(
                detections,
                hidden_states[-1],
                image_sizes,
                box_score_thresh=self.box_score_thresh,
                human_idx=self.human_idx,
                min_instances=self.min_instances,
                max_instances=self.max_instances,
            )
            # PViC's frozen prior operator intentionally stores detector
            # confidence in FP32. Under an outer BF16 autocast H-DETR emits
            # BF16 scores, and indexed assignment into that FP32 tensor is
            # illegal. Proposal geometry/confidence is also a numerical API
            # boundary, so normalize it once before every CoRISP variant uses
            # the proposals; detector hidden states remain autocast-enabled.
            proposals = _normalize_proposal_precision(proposals)

        dense = features[-1].tensors
        dense_valid = ~features[-1].mask
        packets = self.adapter(
            proposals,
            dense,
            image_sizes,
            dense_valid_masks=dense_valid,
            source="frozen_h_detr_swin_l_corisp_only",
        )
        if self.clip_backbone is not None:
            if self.clip_adapter is None:
                raise RuntimeError("A CLIP backbone requires a trainable CLIP pair adapter.")
            clip_tokens = self.clip_backbone(images)
            if clip_tokens.shape[0] != len(packets):
                raise RuntimeError("CLIP and H-DETR returned different batch sizes.")
            packets = [
                self.clip_adapter(packet, clip_tokens[index : index + 1])
                for index, packet in enumerate(packets)
            ]

        boxes: list[Tensor] = []
        pair_indices: list[Tensor] = []
        prior_scores: list[Tensor] = []
        object_types: list[Tensor] = []
        for proposal, packet in zip(proposals, packets):
            if packet.subject_indices is None or packet.object_indices is None:
                raise RuntimeError("The detector-native pair adapter did not retain pair indices.")
            subjects = packet.subject_indices[0]
            objects = packet.object_indices[0]
            pairs = torch.stack([subjects, objects], dim=-1)
            boxes.append(proposal["boxes"])
            pair_indices.append(pairs)
            prior_scores.append(
                compute_prior_scores(
                    subjects,
                    objects,
                    proposal["scores"],
                    proposal["labels"],
                    self.num_verbs,
                    self.training,
                    self.object_to_target,
                )
            )
            object_types.append(proposal["labels"][objects])
        return (
            image_sizes,
            proposals,
            boxes,
            pair_indices,
            prior_scores,
            object_types,
            packets,
        )

    def _predict_packets(
        self, packets: list
    ) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
        logits: list[Tensor] = []
        diagnostics: list[dict[str, Tensor]] = []
        for packet in packets:
            if packet.num_pairs == 0:
                logits.append(packet.pair_feats.new_zeros((0, self.num_verbs)))
                diagnostics.append({"pair_valid_mask": packet.valid_pairs})
                continue
            if self.strong is None:
                logits.append(self.binary_classifier(packet.pair_feats).squeeze(0))
                diagnostics.append({"pair_valid_mask": packet.valid_pairs})
                continue
            outputs = self.strong(packet)
            finalized = self.strong.finalize_joint(
                self.binary_classifier(outputs["interaction_pair_feats"]),
                self.binary_classifier(outputs["enhanced_pair_feats"]),
                outputs,
                self.binary_classifier(outputs["erased_pair_feats"]),
            )
            logits.append(finalized["hoi_logits"].squeeze(0))
            diagnostics.append(finalized)
        return logits, diagnostics

    def _classification_loss(
        self,
        logits: Tensor,
        prior_scores: list[Tensor],
        labels: Tensor,
    ) -> Tensor:
        prior = torch.cat(prior_scores, dim=0).prod(dim=1)
        pair_index, class_index = torch.nonzero(prior, as_tuple=True)
        normalizer = labels.sum()
        if dist.is_initialized():
            normalizer = normalizer.detach().clone()
            dist.all_reduce(normalizer)
            normalizer = normalizer / dist.get_world_size()
        normalizer = normalizer.clamp_min(1.0)
        # The normalizer collective must not depend on whether this rank has a
        # legal local pair; otherwise DDP ranks can enter different all-reduce
        # sequences on sparse V-COCO batches.
        if pair_index.numel() == 0:
            return logits.sum() * 0.0
        selected_logits = logits[pair_index, class_index]
        selected_prior = prior[pair_index, class_index]
        selected_labels = labels[pair_index, class_index]
        calibrated = torch.log(
            selected_prior
            / (1.0 + torch.exp(-selected_logits) - selected_prior)
            + 1e-8
        )
        return binary_focal_loss_with_logits(
            calibrated,
            selected_labels,
            reduction="sum",
            alpha=self.alpha,
            gamma=self.gamma,
        ) / normalizer

    def _participation_loss(
        self,
        diagnostics: list[dict[str, Tensor]],
        prior_scores: list[Tensor],
        labels: Tensor,
    ) -> Tensor:
        if self.strong is None:
            return self.binary_classifier.weight.sum() * 0.0
        available = [
            item["event_participation_logits"].squeeze(0)
            for item in diagnostics
            if "event_participation_logits" in item
        ]
        if not available:
            return self.strong.probability_closure.effective_scale().sum() * 0.0
        logits = torch.cat(available, dim=0)
        valid = torch.cat(prior_scores, dim=0).prod(dim=1) > 0
        if logits.shape != labels.shape or valid.shape != labels.shape:
            raise ValueError("Event participation, labels, and legal-class prior must align.")
        if not valid.any():
            return logits.sum() * 0.0
        return binary_focal_loss_with_logits(
            logits[valid],
            labels[valid].to(logits.dtype),
            alpha=0.25,
            gamma=2.0,
            reduction="mean",
        )

    def _full_joint_loss(
        self,
        diagnostics: list[dict[str, Tensor]],
        boxes: list[Tensor],
        paired_indices: list[Tensor],
        targets: List[dict],
    ) -> Tensor:
        totals: list[Tensor] = []
        metrics: dict[str, list[Tensor]] = {}
        for output, proposal_boxes, pairs, target in zip(
            diagnostics,
            boxes,
            paired_indices,
            targets,
        ):
            if "conditional_role_logits" not in output:
                continue
            aligned = build_prepared_pair_joint_targets(
                output,
                proposal_boxes,
                pairs,
                target,
                self.prepared_target_cfg,
            )
            loss_dict = corisp_joint_loss(output, aligned, self.joint_loss_cfg)
            totals.append(loss_dict["loss_corisp_total"])
            for name, value in loss_dict.items():
                if name == "loss_corisp_total":
                    continue
                metrics.setdefault(name, []).append(value.detach())
        if not totals:
            raise RuntimeError(
                "HEIR full supervision found no proposal pair that can drive the CoRISP joint loss."
            )
        self._last_corisp_metrics = {
            name: torch.stack(values).mean()
            for name, values in metrics.items()
            if values
        }
        return torch.stack(totals).mean()

    def _postprocess(
        self,
        boxes: list[Tensor],
        paired_indices: list[Tensor],
        object_types: list[Tensor],
        logits: list[Tensor],
        prior_scores: list[Tensor],
        image_sizes: Tensor,
    ) -> list[dict[str, Tensor]]:
        outputs: list[dict[str, Tensor]] = []
        for box, pairs, objects, pair_logits, prior, size in zip(
            boxes, paired_indices, object_types, logits, prior_scores, image_sizes
        ):
            legal_prior = prior.prod(dim=1)
            pair_index, class_index = torch.nonzero(legal_prior, as_tuple=True)
            scores = (
                pair_logits[pair_index, class_index].sigmoid()
                * legal_prior[pair_index, class_index].pow(self.raw_lambda)
            )
            outputs.append(
                {
                    "boxes": box,
                    "pairing": pairs[pair_index],
                    "scores": scores,
                    "labels": class_index,
                    "objects": objects[pair_index],
                    "size": size,
                    "x": pair_index,
                }
            )
        return outputs

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        if self.training and targets is None:
            raise ValueError("Training requires dataset targets.")
        (
            image_sizes,
            _,
            boxes,
            pair_indices,
            prior_scores,
            object_types,
            packets,
        ) = self._extract_packets(images)
        logits, diagnostics = self._predict_packets(packets)

        if self.training:
            assert targets is not None
            labels = associate_with_ground_truth(
                boxes, pair_indices, targets, self.num_verbs
            )
            joined_logits = torch.cat(logits, dim=0)
            losses = {
                "cls_loss": self._classification_loss(
                    joined_logits, prior_scores, labels
                )
            }
            if self.strong is not None and self.supervision_mode == "hico_only":
                losses["event_participation_loss"] = (
                    self.participation_loss_weight
                    * self._participation_loss(diagnostics, prior_scores, labels)
                )
            elif self.strong is not None:
                losses["corisp_joint_loss"] = self._full_joint_loss(
                    diagnostics,
                    boxes,
                    pair_indices,
                    targets,
                )
            return losses

        return self._postprocess(
            boxes,
            pair_indices,
            object_types,
            logits,
            prior_scores,
            image_sizes,
        )
