"""Common, GT-free native-slot assembly and V-COCO complete-set AP.

This is a paper-defined extension, not the official V-COCO role metric.
Subject clustering is fixed at inclusive-pixel IoU 0.7 for every model.
"""

from collections import defaultdict
import math

import numpy as np

PROFILE = "vcoco-common-person-clusters-native-slots-v1"
PERSON_IOU = 0.7


def overlaps(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    lo = np.maximum(first[:, None, :2], second[None, :, :2])
    hi = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(hi - lo + 1, 0).prod(-1)
    a = np.maximum(first[:, 2:] - first[:, :2] + 1, 0).prod(-1)
    b = np.maximum(second[:, 2:] - second[:, :2] + 1, 0).prod(-1)
    return intersection / np.maximum(a[:, None] + b[None, :] - intersection, 1e-12)


def box_key(value):
    box = np.asarray(value, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError("Expected four box coordinates")
    if np.isnan(box).all() or (box == 0).all():
        return None
    if not np.isfinite(box).all():
        raise ValueError("Partially nonfinite box")
    return tuple(float(v) for v in box)


def candidate_key(candidate):
    score, box = candidate
    return (-score, box is not None, box if box is not None else ())


def assemble(records, actions, roles):
    slots = [(a, r, f"{action}_{role}") for a, action in enumerate(actions)
             if action != "point" for r, role in enumerate(roles[a][1:])]
    exact = {}
    audit = defaultdict(int, raw_records=len(records))
    for record in records:
        person = box_key(record["person_box"])
        if person is None:
            raise ValueError("Subject cannot be null")
        selected = exact.setdefault(person, {})
        for a, r, field in slots:
            if field not in record:
                continue
            value = np.asarray(record[field], dtype=np.float32)
            if value.shape != (5,):
                raise ValueError("Expected role box and score")
            score = float(value[4])
            if math.isnan(score):
                audit["nan_score_fields_omitted"] += 1
                continue
            if not math.isfinite(score):
                raise ValueError("Infinite score")
            audit["score_fields_outside_unit_interval"] += int(not 0 <= score <= 1)
            candidate = (score, box_key(value[:4]))
            if (a, r) not in selected or candidate_key(candidate) < candidate_key(selected[a, r]):
                selected[a, r] = candidate
    # Representatives never move: membership is not transitive/chain merging.
    ordered = sorted(exact, key=lambda b: (-max((v[0] for v in exact[b].values()), default=-math.inf), b))
    representatives, groups = [], []
    for person in ordered:
        quality = overlaps([person], representatives)[0]
        best = int(quality.argmax()) if len(quality) else -1
        if best < 0 or quality[best] < PERSON_IOU:
            representatives.append(person)
            groups.append({})
            best = len(groups) - 1
        for slot, candidate in exact[person].items():
            if slot not in groups[best] or candidate_key(candidate) < candidate_key(groups[best][slot]):
                groups[best][slot] = candidate
    events = []
    for person, selected in zip(representatives, groups):
        for a, action in enumerate(actions):
            count = len(roles[a]) - 1
            if not count or action == "point":
                continue
            values = [selected.get((a, r)) for r in range(count)]
            if any(value is None for value in values):
                continue
            events.append({"person": person, "action": a, "entities": [v[1] for v in values],
                           "score": min(v[0] for v in values)})
    audit.update(exact_person_boxes=len(exact), person_clusters=len(groups), event_hypotheses=len(events))
    return events, dict(audit)


def score_image(entry, events, roles):
    persons = np.flatnonzero(entry["gt_classes"] == 1)
    nodes = sorted({event["person"] for event in events})
    matrix = overlaps(nodes, entry["boxes"][persons])
    nearest = {}
    for i, node in enumerate(nodes):
        if len(persons):
            j = int(matrix[i].argmax())
            nearest[node] = (int(persons[j]), float(matrix[i, j]))
    outcomes, covered = [], [set(), set()]
    ignored = 0
    for event in sorted(events, key=lambda e: (-e["score"], e["person"], e["action"])):
        person, iou = nearest.get(event["person"], (-1, 0.0))
        # Match the official nearest-person ignore convention, including low IoU.
        if person >= 0 and np.all(entry["gt_actions"][person] == -1):
            ignored += 1
            continue
        action = event["action"]
        valid = person >= 0 and iou >= .5 and entry["gt_actions"][person, action] == 1
        correct = [bool(valid), bool(valid)]
        for slot, entity in enumerate(event["entities"]):
            target = int(entry["gt_role_id"][person, action, slot]) if valid else -2
            if target == -1:
                correct[0] &= entity is None
            elif target < 0 or entity is None:
                correct = [False, False]
            else:
                hit = bool(overlaps([entity], entry["boxes"][target:target + 1])[0, 0] >= .5)
                correct = [c and hit for c in correct]
        key = (person, action)
        for s in range(2):
            correct[s] &= key not in covered[s]
            if correct[s]:
                covered[s].add(key)
        outcomes.append((action, event["score"], *correct))
    return outcomes, ignored


def average_precision(rows, positives):
    if positives <= 0:
        return None
    if not rows:
        return 0.0
    rows = sorted(rows, key=lambda v: -v[0])
    scores = np.asarray([r[0] for r in rows])
    correct = np.asarray([r[1] for r in rows], dtype=np.float64)
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(rows) - 1]
    tp = np.cumsum(correct)[ends]
    if tp[-1] > positives:
        raise ValueError("More true positives than ground-truth sets")
    precision = np.maximum.accumulate((tp / (ends + 1))[::-1])[::-1]
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(database, by_image, actions, roles):
    ids = [int(e["id"]) for e in database]
    if len(ids) != len(set(ids)) or not set(by_image) <= set(ids):
        raise ValueError("Invalid image coverage")
    support, outcomes, audit = defaultdict(int), defaultdict(list), defaultdict(int)
    for index, entry in enumerate(sorted(database, key=lambda e: int(e["id"]))):
        events, counts = assemble(by_image.get(int(entry["id"]), []), actions, roles)
        for key, value in counts.items():
            audit[key] += value
        scored, ignored = score_image(entry, events, roles)
        audit["ignored_hypotheses"] += ignored
        for person in np.flatnonzero(entry["gt_classes"] == 1):
            for a, action in enumerate(actions):
                if action != "point" and len(roles[a]) > 1 and entry["gt_actions"][person, a] == 1:
                    support[a] += 1
        for a, score, s1, s2 in scored:
            outcomes[a, 0].append((score, s1))
            outcomes[a, 1].append((score, s2))
        if (index + 1) % 1000 == 0:
            print(f"Scored {index + 1}/{len(database)} images", flush=True)
    results = {}
    for s in range(2):
        rows = [{"action": actions[a], "slots": len(roles[a]) - 1, "gt_sets": support[a],
                 "ap_percent": 100 * average_precision(outcomes[a, s], support[a])} for a in sorted(support)]
        results[f"scenario_{s + 1}"] = {"set_map_percent": float(np.mean([r["ap_percent"] for r in rows])),
            "dual_slot_map_percent": float(np.mean([r["ap_percent"] for r in rows if r["slots"] == 2])),
            "single_slot_map_percent": float(np.mean([r["ap_percent"] for r in rows if r["slots"] == 1])),
            "per_action": rows}
    return {"profile": PROFILE, "images": len(ids), "actions": len(support), "gt_sets": sum(support.values()),
            "audit": dict(audit), "results": results}
