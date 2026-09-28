from __future__ import annotations

DEFAULT_ROLES = (
    "target",
    "instrument",
    "support",
    "container",
    "location",
    "source",
    "destination",
    "constraint",
)

ROLE_ALIASES = {
    "patient": "target",
    "object": "target",
    "tool": "instrument",
    "support_of_agent": "support",
    "support_of_target": "support",
    "support_of_event": "support",
    "ground": "support",
    "carrier": "container",
    "destination_container": "container",
    "location_surface": "location",
    "location_of_event": "location",
    "surface": "location",
    "place": "location",
    "origin": "source",
    "recipient": "destination",
    "endpoint": "destination",
    "regulator": "constraint",
    "barrier": "constraint",
    "controller": "constraint",
}

def normalize_role_name(name: str) -> str:
    key = str(name).strip()
    return ROLE_ALIASES.get(key, key)
