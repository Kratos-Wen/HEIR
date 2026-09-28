"""GT-free JSON event parsing and native role export for the supervised adaptation."""

import json
import math


def reject_constant(value):
    raise ValueError("Non-finite JSON value: " + value)


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def restore_box(box, geometry):
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 <= v <= 1000 for v in box)):
        raise ValueError("Box must have four finite coordinates in 0..1000")
    side = geometry["padded_side"]
    left, top = geometry["pad_left_top"]
    width, height = geometry["original_wh"]
    x1, y1, x2, y2 = [v * side / 1000 - shift for v, shift in zip(box, [left, top, left, top])]
    x1, x2 = max(0., x1), min(float(width), x2)
    y1, y2 = max(0., y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Degenerate or wholly padded-region box")
    # Training uses x+w/y+h. The official evaluator uses inclusive x+w-1/y+h-1.
    return [min(width - 1., x1), min(height - 1., y1),
            max(x1, x2 - 1.), max(y1, y2 - 1.)]


def action_roles(channels):
    actions = {}
    for channel in channels:
        action, role = channel.rsplit("_", 1)
        actions.setdefault(action, [])
        if role != "agent":
            actions[action].append(role)
    return actions


def event_spans(text):
    decoder = json.JSONDecoder(object_pairs_hook=unique_keys, parse_constant=reject_constant)
    value = decoder.decode(text)
    if not isinstance(value, list):
        raise ValueError("Expected an event list, not a prose answer")
    cursor = text.index("[") + 1
    result = []
    for event in value:
        while cursor < len(text) and text[cursor] in " \t\n\r,":
            cursor += 1
        parsed, end = decoder.raw_decode(text, cursor)
        if parsed != event:
            raise ValueError("Event span mismatch")
        result.append((event, cursor, end))
        cursor = end
    return result


def parse_prediction(text, geometry, image_id, actions, logprobs, token_spans=None):
    """Return valid hypotheses and explicit errors; never invent missing slots."""
    errors, output = [], []
    try:
        spans = event_spans(text)
    except (ValueError, TypeError) as error:
        return [], [{"scope": "image", "reason": str(error)}]
    if not logprobs or not all(math.isfinite(v) and v <= 1e-6 for v in logprobs):
        return [], [{"scope": "image", "reason": "Invalid generated-token log probabilities"}]
    for index, (event, begin, end) in enumerate(spans):
        try:
            if not isinstance(event, dict) or set(event) != {"human", "action", "roles"}:
                raise ValueError("Invalid event keys")
            action = event["action"]
            if not isinstance(action, str) or action not in actions:
                raise ValueError("Unknown native action")
            if not isinstance(event["roles"], dict) or set(event["roles"]) != set(actions[action]):
                raise ValueError("Missing/extra role is not an explicitly predicted null")
            person = restore_box(event["human"], geometry)
            roles = {}
            for role, entity in event["roles"].items():
                if entity is None:
                    roles[role] = None
                    continue
                if not isinstance(entity, dict) or set(entity) != {"box", "noun_id"}:
                    raise ValueError("Invalid role entity")
                noun = entity["noun_id"]
                if isinstance(noun, bool) or not isinstance(noun, int) or not 0 <= noun < 80:
                    raise ValueError("Invalid COCO noun id")
                roles[role] = {"box": restore_box(entity["box"], geometry), "noun_id": noun}
            chosen = ([v for v, (a, b) in zip(logprobs, token_spans) if b > begin and a < end]
                      if token_spans is not None else logprobs)
            if not chosen:
                raise ValueError("No model confidence for event")
            score = math.exp(sum(chosen) / len(chosen))
            output.append({"image_id": image_id, "person_box": person, "action": action,
                           "roles": roles, "score": score, "generation_event_index": index})
        except (ValueError, TypeError, KeyError) as error:
            errors.append({"scope": "event", "event_index": index, "reason": str(error)})
    return output, errors


def native_records(events, actions):
    records = []
    for event in events:
        row = {"image_id": event["image_id"], "person_box": event["person_box"]}
        # Official NaN scores omit unpredicted classes; zero scores could fabricate
        # extra action hypotheses from an unrelated generated event.
        for action, roles in actions.items():
            row[action + "_agent"] = math.nan
            for role in roles:
                row[action + "_" + role] = [math.nan] * 5
        action, score = event["action"], event["score"]
        row[action + "_agent"] = score
        for role, entity in event["roles"].items():
            row[action + "_" + role] = ([0.] * 4 if entity is None else entity["box"]) + [score]
        records.append(row)
    return records
