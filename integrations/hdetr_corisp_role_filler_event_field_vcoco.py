from __future__ import annotations

from typing import List, Optional

from torch import Tensor

from hdetr_corisp_grounded_role_set_vcoco import HDETRCoRISPGroundedRoleSetVCOCO

from corisp import ROLE_FILLER_EVENT_FIELD_INTERVENTIONS, ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS, RoleFillerEventField

class HDETRCoRISPRoleFillerEventFieldVCOCO(HDETRCoRISPGroundedRoleSetVCOCO):
    """Predict each person-predicate event as one exact role-filler set."""

    @property
    def role_filler_event_field(self) -> RoleFillerEventField:
        field = self.grounded_role_set
        if not isinstance(field, RoleFillerEventField):
            raise TypeError("The role-filler adapter requires RoleFillerEventField.")
        return field

    @property
    def role_filler_intervention(self) -> str:
        value = getattr(self, "_role_filler_intervention", "full")
        if value not in ROLE_FILLER_EVENT_FIELD_INTERVENTIONS:
            raise ValueError(f"Invalid role-filler intervention {value!r}.")
        return value

    @property
    def role_filler_structural_ablation(self) -> str:
        value = getattr(self, "_role_filler_structural_ablation", "full")
        if value not in ROLE_FILLER_EVENT_FIELD_STRUCTURAL_ABLATIONS:
            raise ValueError(
                f"Invalid role-filler structural ablation {value!r}."
            )
        return value

    def _detector_composed_unaries(
        self,
        output: dict[str, Tensor],
        proposal: dict[str, Tensor],
        role_class_prior: Tensor,
    ) -> tuple[Tensor, Tensor]:
        class_prior = role_class_prior.prod(dim=1)
        role_prior = class_prior.new_zeros(
            (
                class_prior.shape[0],
                self.role_space.num_actions,
                self.role_space.num_roles,
            )
        )
        for class_index, (action_index, role_index) in enumerate(
            zip(
                self.role_space.role_class_to_action,
                self.role_space.role_class_to_role,
            )
        ):
            role_prior[:, action_index, role_index] = class_prior[:, class_index]

        visible = output["joint_role_probs"].squeeze(0) * role_prior
        agent_entities = output["agent_entity_indices"].squeeze(0).long()
        typed = output["null_role_probs"].squeeze(0)[agent_entities]
        typed = typed * proposal["scores"][agent_entities, None, None]
        return visible, typed

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
        distribution = self.role_filler_event_field(
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
            intervention=self.role_filler_intervention,
            structural_ablation=self.role_filler_structural_ablation,
        )
        output["role_filler_refinement_scale"] = distribution.refinement_scale
        output["role_signature_scale"] = distribution.signature_scale
        return distribution

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None,
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        if self.training and targets is None:
            raise ValueError("Role-filler event-field training requires targets.")
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
                "role_filler_event_field_loss": self._structured_event_loss(
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
