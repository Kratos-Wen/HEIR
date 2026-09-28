from __future__ import annotations

import os

from pathlib import Path

from typing import List, Optional

import torch

from torch import Tensor

from h_detr.models import build_model as build_advanced_detr

from dinotxt_event_adapter import FrozenDINOtxtVisualBackbone, load_dinotxt_prototype_bank, load_dinotxt_role_prototype_bank

from hdetr_corisp_role_filler_event_field_vcoco import HDETRCoRISPRoleFillerEventFieldVCOCO

from hdetr_corisp_v6_vcoco import _JointStateClassifierContract

from corisp import PreparedProposalAdapterConfig, PreparedProposalPairAdapter, CoRISPAgentRoleField, CoRISPAgentRoleFieldConfig, RoleArityEventField, RoleArityEventFieldConfig, VCOCORoleSpace

class HDETRCoRISPRoleArityEventFieldVCOCO(
    HDETRCoRISPRoleFillerEventFieldVCOCO
):
    """Predict a learned 0/1/2+ occupancy state for every event role."""

    @property
    def role_filler_event_field(self) -> RoleArityEventField:
        field = self.grounded_role_set
        if not isinstance(field, RoleArityEventField):
            raise TypeError("The role-count adapter requires RoleArityEventField.")
        return field

    @property
    def role_arity_event_field(self) -> RoleArityEventField:
        return self.role_filler_event_field

    def _role_filler_distribution(
        self,
        *,
        output: dict[str, Tensor],
        proposal: dict[str, Tensor],
        role_class_prior: Tensor,
        packet,
        return_marginals: bool,
    ):
        visible, typed = self._detector_composed_unaries(
            output, proposal, role_class_prior
        )
        dtype = output["edge_states"].dtype
        device = output["edge_states"].device
        action_anchor = self.role_field.semantic_projection(
            self.role_field.action_prototypes.to(device=device, dtype=dtype)
        )
        role_anchor = self.role_field.semantic_projection(
            self.role_field.role_prototypes.to(device=device, dtype=dtype)
        ) + self.role_field.role_offset.to(device=device, dtype=dtype)
        object_indices = packet.object_indices.squeeze(0).long()
        object_labels = packet.entity_labels.squeeze(0)[object_indices].long()
        noun_anchor = self.role_field.semantic_projection(
            self.role_field.object_prototypes[object_labels].to(
                device=device, dtype=dtype
            )
        )
        agent_entities = output["agent_entity_indices"].squeeze(0).long()
        distribution = self.role_arity_event_field(
            base_visible_role_probs=visible,
            base_typed_null_role_probs=typed,
            edge_states=output["edge_states"].squeeze(0),
            null_edge_states=output["null_edge_states"].squeeze(0)[agent_entities],
            event_states=output["event_states"].squeeze(0),
            pair_agent_indices=output["pair_agent_indices"].squeeze(0).long(),
            pair_entity_indices=object_indices,
            noun_anchors=noun_anchor,
            action_anchors=action_anchor,
            role_anchors=role_anchor,
            visible_action_mask=output["visible_action_mask"].squeeze(0).bool(),
            valid_role_mask=self.role_space.valid_role_mask().to(device=device),
            return_marginals=return_marginals,
        )
        output["role_filler_refinement_scale"] = distribution.refinement_scale
        output["role_arity_scale"] = distribution.arity_scale
        return distribution

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        if self.training and targets is None:
            raise ValueError("Role-arity event-field training requires targets.")
        (
            image_sizes,
            proposals,
            boxes,
            pair_indices,
            _,
            object_types,
            packets,
        ) = self._extract_packets(images)
        outputs = self._predict_role_packets(images, packets)
        role_class_priors = self._role_class_priors(proposals, pair_indices)
        distributions = [
            self._role_filler_distribution(
                output=output,
                proposal=proposal,
                role_class_prior=prior,
                packet=packet,
                return_marginals=not self.training,
            )
            for output, proposal, prior, packet in zip(
                outputs, proposals, role_class_priors, packets
            )
        ]

        if self.training:
            assert targets is not None
            constraints = [
                self._event_target_groups(
                    distribution=distribution,
                    output=output,
                    proposal=proposal,
                    pairs=pairs,
                    target=target,
                )
                for distribution, output, proposal, pairs, target in zip(
                    distributions, outputs, proposals, pair_indices, targets
                )
            ]
            return {
                "role_arity_event_field_loss": self._structured_event_loss(
                    distributions, constraints
                )
            }

        return self._postprocess_grounded_role_sets(
            boxes,
            proposals,
            pair_indices,
            object_types,
            outputs,
            distributions,
            role_class_priors,
            image_sizes,
        )

def build_hdetr_corisp_role_arity_event_field_vcoco(
    args,
    dataset,
) -> HDETRCoRISPRoleArityEventFieldVCOCO:
    role_space = VCOCORoleSpace.from_role_classes(
        dataset.actions, dataset.num_instances
    )
    object_to_role_class = list(dataset.object_to_action.values())
    args.num_verbs = role_space.num_role_classes
    detector, _, postprocessors = build_advanced_detr(args)
    if not os.path.isfile(args.pretrained):
        raise FileNotFoundError(f"Missing H-DETR checkpoint: {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    detector.load_state_dict(checkpoint["model_state_dict"])

    d_model = int(args.repr_dim)
    backbone_channels = detector.backbone.num_channels
    dense_dim = int(
        backbone_channels[-1]
        if isinstance(backbone_channels, list)
        else backbone_channels
    )
    adapter = PreparedProposalPairAdapter(
        PreparedProposalAdapterConfig(
            detector_dim=int(args.hidden_dim),
            pair_dim=d_model,
            dense_dim=dense_dim,
            d_model=d_model,
            human_label=0,
            max_pairs=max(
                1024,
                int(args.max_instances) * (2 * int(args.max_instances) - 1),
            ),
            include_human_human=True,
        )
    )
    action_prototypes, object_prototypes, _ = load_dinotxt_prototype_bank(
        Path(os.environ["CORISP_DINOTXT_PROTOTYPES"]),
        action_names=role_space.action_names,
        object_names=dataset.objects[1:],
    )
    role_prototypes, _ = load_dinotxt_role_prototype_bank(
        Path(os.environ["CORISP_DINOTXT_ROLE_PROTOTYPES"]),
        role_names=role_space.role_names,
    )
    semantic_backbone = FrozenDINOtxtVisualBackbone(
        backbone_weights=os.environ["CORISP_DINOV3_BACKBONE"],
        dinotxt_weights=os.environ["CORISP_DINOTXT_WEIGHTS"],
        input_size=int(os.environ.get("CORISP_DINO_INPUT_SIZE", "448")),
        verify_hashes=os.environ.get("CORISP_SKIP_DINO_HASH", "0") != "1",
    )
    role_field = CoRISPAgentRoleField(
        CoRISPAgentRoleFieldConfig(
            d_model=d_model,
            visual_dim=semantic_backbone.output_dim,
            semantic_dim=semantic_backbone.semantic_dim,
            num_actions=role_space.num_actions,
            roles=role_space.role_names,
            num_object_classes=len(dataset.objects) - 1,
            inference_steps=int(os.environ.get("CORISP_FIELD_STEPS", "2")),
            ffn_dim=d_model * 4,
            dropout=0.1,
            enable_typed_null_fillers=True,
        ),
        action_prototypes=action_prototypes,
        object_prototypes=object_prototypes,
        role_prototypes=role_prototypes,
        valid_role_mask=role_space.valid_role_mask(),
        object_action_mask=role_space.observed_object_action_mask(
            object_to_role_class
        ),
        human_label=0,
    )
    event_field = RoleArityEventField(
        RoleArityEventFieldConfig(
            d_model=d_model,
            num_roles=role_space.num_roles,
            max_cardinality=(2 * int(args.max_instances) - 1)
            + role_space.num_roles,
            num_heads=int(os.environ.get("CORISP_ROLE_ARITY_HEADS", "8")),
            arity_rank=int(os.environ.get("CORISP_ROLE_ARITY_RANK", "64")),
            dropout=float(os.environ.get("CORISP_ROLE_ARITY_DROPOUT", "0.1")),
            max_log_residual=float(
                os.environ.get("CORISP_ROLE_ARITY_MAX_LOG_RESIDUAL", "4.0")
            ),
            max_arity_energy=float(
                os.environ.get("CORISP_ROLE_ARITY_MAX_ENERGY", "4.0")
            ),
            max_arity_states=int(
                os.environ.get("CORISP_ROLE_ARITY_MAX_STATES", "8192")
            ),
        )
    )
    model = HDETRCoRISPRoleArityEventFieldVCOCO(
        detector=detector,
        postprocessor=postprocessors["bbox"],
        adapter=adapter,
        classifier=_JointStateClassifierContract(
            d_model, role_space.num_role_classes
        ),
        object_to_target=object_to_role_class,
        strong=None,
        variant="hdetr_corisp_role_arity_event_field",
        human_idx=0,
        box_score_thresh=float(args.box_score_thresh),
        min_instances=int(args.min_instances),
        max_instances=int(args.max_instances),
        raw_lambda=1.0,
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        participation_loss_weight=0.0,
        supervision_mode="vcoco_roles",
        role_space=role_space,
        object_to_role_class=object_to_role_class,
        role_field=role_field,
        semantic_backbone=semantic_backbone,
        null_role_loss_weight=float(args.vcoco_role_loss_weight),
        grounded_role_set=event_field,
    )
    model.freeze_detector()
    return model
