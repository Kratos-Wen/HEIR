"""CoRISP model components and proposal interfaces."""
from corisp.event_set_field import EventSet, EventSetField, EventSetFieldConfig, EventSetOutput, TargetGroup
from corisp.interaction_field import InteractionFieldConfig, RoleConditionedInteractionField
from corisp.pair_packet import PairPacket, ProposalPairAdapter, ProposalPairAdapterConfig
from corisp.vcoco_nulls import VCOCONullRoleIndex, VCOCONullRoleRecord, load_vcoco_null_role_index
from corisp.vcoco_roles import VCOCORoleSpace
