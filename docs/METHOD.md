# CoRISP: paper-to-code correspondence

| Paper operation | Implementation |
| --- | --- |
| Joint null-plus-role logits and role feedback | `CoRISPAgentRoleField._joint_state`, `CoRISPAgentRoleField.forward` |
| Two tied recurrent updates | `CoRISPAgentRoleFieldConfig.inference_steps=2` |
| Event, pair and shared-entity summaries | `RoleFillerEventField._relational_unaries` |
| Detached role weights and same-role self exclusion | `RoleFillerEventField._relational_unaries` |
| Attention to role summaries | `RoleFillerEventField._attend_source_roles` |
| Relation fusion | `RoleFillerEventField._combine_relations` |
| Additive/multiplicative binding and event role context | `RoleFillerEventField._relational_unaries` |
| Refined positive-role probabilities | `RoleFillerEventField._refine_visible` |
| Cardinality B and saturated role-multiplicity G | `RoleArityEventField._event_composition_energy` |
| Exact event partition and marginals | `RoleArityEventField`, `corisp_heir.batched_dp`, `corisp_heir.adjoint_dp` |
| Unique compatible localization assignments | `RoleArityEventField.target_log_mass` |
| Fixed compatibility support | `corisp_heir.support` |
| Packed FP32 potentials | `corisp_heir_bucket.packed`, `corisp_heir_bucket.vectorized` |

Each candidate is either unselected or assigned one role. Different candidates may fill the same role, and an entity may participate in multiple events. The role-count statistic distinguishes absence, a single participant and repetition (0/1/2+), while total cardinality retains the event size. Normalization is exact within one event's retained candidate and support space. Events share features and have separate normalizers. Recurrent feedback uses preliminary local probabilities; the set posterior is computed after context aggregation.

The model uses weighted-mean role summaries. The cardinality potential B and role-multiplicity potential G enter the same assignment energy and share its partition function.

On HEIR, the local field and role-composition pooling use all six functional roles. The support mask is applied at set inference. Compact DP states omit inactive role axes while retaining their count-zero contribution to G, including the six-role pooling normalization. On V-COCO, the native action-role mask determines the active slots.

The three component controls change training, not decoding: `no_arity` removes G while retaining B; `no_relations` removes contextual messages; `no_role_feedback` uses interaction-only recurrent weights and omits the conditional role vector. All retain the final null-plus-role predictor and the same set prediction rule.

The visual-candidate distribution is categorical over null plus roles. V-COCO also has one missing-filler candidate for each native role, with a binary distribution from `CoRISPAgentRoleField._typed_null_state`; these candidates do not compete across roles. They receive event context, but not visible pair/entity context.

The model's interaction probability sums positive-role marginals. HEIR's benchmark HOI projection instead takes the maximum role score for each person--entity--action tuple, identically for all evaluated systems.

HEIR set prediction ranks one MAP assignment per reachable count state and keeps up to eight state winners per event, followed by a 100-set image budget. Each state contributes one assignment even when several assignments share its counts. `evaluation.native.decode_state_winners` implements this rule. V-COCO evaluates its fixed native slots, as specified in the paper.

## Complete-set evaluation

HEIR evaluates the submitted sets with a shared, noun-compatible, one-to-one entity correspondence per image at IoU >= 0.5. Entity priority is the maximum score of a submitted set containing it. Ties are resolved by entity confidence and then entity ID. Each entity matches the unused same-noun ground-truth entity of highest IoU, with ground-truth ID resolving equal IoUs. A true positive must recover the correct actor, action and complete entity--role membership, and each annotated event is matched at most once. Equal-score outcomes are grouped before computing the all-point interpolated precision envelope; Set mAP averages AP over supported actions. `evaluation.heir.core` implements matching and AP, and `evaluation.heir_sets` handles the prediction format and annotation scope.
