"""Convert HEIR ground truth and relation predictions to the heir_eval input format.

GT: annotations/{split}.json. Classes are the (verb, noun, role) tuples present in the scored split, with
train_count from annotations/train.json (unseen 0, rare 1-9, non-rare >=10). Events are not annotated, so
events_complete is False and only relation AP is meaningful.

Predictions: one record per image with its entities (nodes) and scored (subject, entity, verb, role)
edges (`relations`).

agent_only rule: on SWiG-origin test images only predictions whose subject matches an annotated agent
(IoU >= 0.5) and whose verb equals that agent's annotated verb are scored; every other prediction on such an
image is ignored. heir_eval has no ignore masks, so the rule is applied by removing the ignored predictions
before scoring and reporting how many were removed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heir_data import HEIR_ROLE_NAMES  # noqa: E402

FORMAT = "heir-eval"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def build_ground_truth(root, split, vocab, train_images, split_images):
    train_count = Counter((vocab.verbs[v], vocab.objects[img.categories[o]], vocab.role_names[r])
                          for img in train_images for _, o, v, r in img.relations)
    observed, images = set(), []
    for img in split_images:
        entities = [{"id": f"box_{i + 1}", "noun": vocab.objects[c], "bbox": list(map(float, box))}
                    for i, (box, c) in enumerate(zip(img.boxes, img.categories))]
        relations = []
        for k, (s, o, v, r) in enumerate(img.relations):
            key = (vocab.verbs[v], vocab.objects[img.categories[o]], vocab.role_names[r])
            observed.add(key)
            relations.append({"id": f"rel_{k}", "subject_id": f"box_{s + 1}", "entity_id": f"box_{o + 1}",
                              "verb": key[0], "role": key[2]})
        images.append({"id": img.image_id, "width": img.width, "height": img.height,
                       "cluster_id": img.source or "unknown", "coverage": "adjudicated_closed_world",
                       "events_complete": False, "entities": entities, "relations": relations, "events": []})
    classes = [{"verb": v, "noun": n, "role": r, "train_count": int(train_count[(v, n, r)])}
               for v, n, r in sorted(observed)]
    return {"schema_version": FORMAT, "annotation_policy_id": f"HEIR {vocab.version} {split} native relations",
            "taxonomy": {"verbs": list(vocab.verbs), "nouns": list(vocab.objects), "roles": list(HEIR_ROLE_NAMES)},
            "classes": classes, "images": images}


def build_predictions(records, vocab, split_images):
    by_id = {img.image_id: img for img in split_images}
    images, stats = [], Counter()
    for record in records:
        img = by_id[record["image_id"]]
        width, height = float(img.width), float(img.height)
        entities, valid_node = [], {}
        for node in record["nodes"]:
            x1, y1, x2, y2 = node["box"]
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if not (x1 < x2 and y1 < y2):
                stats["degenerate_nodes"] += 1
                continue
            valid_node[node["id"]] = f"node_{node['id']}"
            entities.append({"id": valid_node[node["id"]], "noun": vocab.objects[int(node["category_id"])],
                             "bbox": [x1, y1, x2, y2], "score": float(node["detector_score"])})
        boxes = {e["id"]: e["bbox"] for e in entities}
        # agent_only: agents and verbs that may be scored on this image.
        agent_verbs = None
        if img.evaluation == "agent_only":
            agent_verbs = {}
            for s, _, v, _ in img.relations:
                agent_verbs.setdefault(s, set()).add(vocab.verbs[v])
        relations = []
        for edge in record["relations"]:
            if edge["subject"] not in valid_node or edge["entity"] not in valid_node:
                stats["skipped_null_or_invalid_edges"] += 1
                continue
            subject_id, entity_id = valid_node[edge["subject"]], valid_node[edge["entity"]]
            if agent_verbs is not None:
                subject_box = boxes[subject_id]
                scored = any(iou(subject_box, img.boxes[a]) >= 0.5 and edge["verb"] in verbs
                             for a, verbs in agent_verbs.items())
                if not scored:
                    stats["agent_only_ignored_predictions"] += 1
                    continue
            relation_id = f"rel_{len(relations)}"
            relations.append({"id": relation_id, "subject_id": subject_id, "entity_id": entity_id,
                              "verb": edge["verb"], "role": edge["role"], "score": float(edge["score"])})
        images.append({"id": img.image_id, "entities": entities, "relations": relations, "events": []})
        stats["images"] += 1
        stats["relations"] += len(relations)
    return {"schema_version": FORMAT, "images": images}, dict(stats)
