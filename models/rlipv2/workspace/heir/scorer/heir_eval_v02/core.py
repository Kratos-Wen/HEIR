from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import ROLES, VERSION, class_key, finite_number, relation_key, require, validate_inputs


def mean_or_none(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def box_iou(first, second):
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def average_precision(outcomes, positive_count):
    require(isinstance(positive_count, int) and positive_count >= 0, "invalid AP positive count")
    if positive_count == 0:
        return None
    grouped = defaultdict(lambda: [0, 0])
    for score, correct in outcomes:
        require(finite_number(score) and isinstance(correct, bool), "invalid AP observation")
        grouped[float(score)][0 if correct else 1] += 1
    if not grouped:
        return 0.0
    counts = np.asarray([grouped[score] for score in sorted(grouped, reverse=True)], dtype=float)
    cumulative = np.cumsum(counts, axis=0)
    require(cumulative[-1, 0] <= positive_count, "TP count exceeds GT count")
    recall = cumulative[:, 0] / positive_count
    precision = cumulative[:, 0] / np.sum(cumulative, axis=1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def empty_prediction(image_id):
    return {"id": image_id, "entities": [], "relations": [], "events": []}


def relation_ap(gt_images, pred_images, classes, iou_threshold):
    references = defaultdict(list)
    predicted = defaultdict(list)
    for image_id, image in gt_images.items():
        entities = {entity["id"]: entity for entity in image["entities"]}
        for relation in image["relations"]:
            references[class_key(relation, entities)].append((image_id, relation, entities))
        prediction = pred_images.get(image_id, empty_prediction(image_id))
        pred_entities = {entity["id"]: entity for entity in prediction["entities"]}
        for relation in prediction["relations"]:
            predicted[class_key(relation, pred_entities)].append((image_id, relation, pred_entities))
    records = []
    for labels, train_count in sorted(classes.items()):
        candidates = defaultdict(list)
        for image_id, relation, entities in references[labels]:
            candidates[image_id].append((relation, entities))
        covered = set()
        outcomes = []
        for image_id, relation, entities in sorted(
                predicted[labels], key=lambda row: (-row[1]["score"], row[0], row[1]["id"])):
            best_overlap, best_id = -1.0, None
            if entities[relation["subject_id"]]["noun"] == "person":
                for reference, ref_entities in sorted(candidates[image_id], key=lambda row: row[0]["id"]):
                    overlap = min(
                        box_iou(entities[relation["subject_id"]]["bbox"], ref_entities[reference["subject_id"]]["bbox"]),
                        box_iou(entities[relation["entity_id"]]["bbox"], ref_entities[reference["entity_id"]]["bbox"]),
                    )
                    if overlap > best_overlap:
                        best_overlap, best_id = overlap, reference["id"]
            correct = best_overlap >= iou_threshold and (image_id, best_id) not in covered
            outcomes.append((relation["score"], bool(correct)))
            if correct:
                covered.add((image_id, best_id))
        records.append({
            "verb": labels[0], "noun": labels[1], "role": labels[2], "train_count": train_count,
            "gt_count": len(references[labels]), "prediction_count": len(predicted[labels]),
            "ap": average_precision(outcomes, len(references[labels])),
        })
    return {
        "mAP": mean_or_none(record["ap"] for record in records),
        "rare_mAP": mean_or_none(record["ap"] for record in records if 1 <= record["train_count"] < 10),
        "non_rare_mAP": mean_or_none(record["ap"] for record in records if record["train_count"] >= 10),
        "unseen_mAP": mean_or_none(record["ap"] for record in records if record["train_count"] == 0),
        "per_role": {
            role: {"mAP": mean_or_none(record["ap"] for record in records if record["role"] == role),
                   "gt_count": sum(record["gt_count"] for record in records if record["role"] == role)}
            for role in ROLES
        },
        "unsupported_prediction_count": sum(len(rows) for key, rows in predicted.items() if key not in classes),
        "classes": records,
    }


def instance_correspondence(reference, prediction, iou_threshold):
    ref_entities = {entity["id"]: entity for entity in reference["entities"]}
    pred_entities = {entity["id"]: entity for entity in prediction["entities"]}
    pred_relations = {relation["id"]: relation for relation in prediction["relations"]}
    priorities = {}
    for event in prediction.get("events", []):
        for relation_id in event["relation_ids"]:
            relation = pred_relations[relation_id]
            for endpoint in (relation["subject_id"], relation["entity_id"]):
                priorities[endpoint] = max(priorities.get(endpoint, -1), event["score"])
    mapping, used = {}, set()
    order = sorted(priorities, key=lambda entity_id: (-priorities[entity_id], -pred_entities[entity_id]["score"], entity_id))
    for entity_id in order:
        entity = pred_entities[entity_id]
        candidates = [
            (box_iou(entity["bbox"], candidate["bbox"]), reference_id)
            for reference_id, candidate in ref_entities.items()
            if reference_id not in used and candidate["noun"] == entity["noun"]
        ]
        if not candidates:
            continue
        overlap, reference_id = sorted(candidates, key=lambda pair: (-pair[0], pair[1]))[0]
        if overlap >= iou_threshold:
            mapping[entity_id] = reference_id
            used.add(reference_id)
    return mapping


def event_edges(event, relations, mapping=None):
    edges = set()
    for relation_id in event["relation_ids"]:
        relation = relations[relation_id]
        subject, entity, verb, role = relation_key(relation)
        if mapping is not None:
            if subject not in mapping or entity not in mapping:
                continue
            subject, entity = mapping[subject], mapping[entity]
        edges.add((subject, entity, verb, role))
    return edges


def structural_stratum(edges, entities):
    people = {edge[0] for edge in edges}
    objects = {edge[1] for edge in edges if entities[edge[1]]["noun"] != "person"}
    if not objects:
        return "human_human"
    return ("1" if len(people) == 1 else "N") + ":" + ("1" if len(objects) == 1 else "M")


def event_metrics(gt_images, pred_images, iou_threshold, event_threshold):
    complete = {image_id: image for image_id, image in gt_images.items() if image["events_complete"]}
    if not complete:
        return {"status": "unavailable_no_complete_event_gt", "evaluated_images": 0,
                "mAP": None, "aligned_macro_F1": None}
    references = defaultdict(list)
    predictions = defaultdict(list)
    counts = defaultdict(lambda: {"correct": 0, "predicted": 0, "gold": 0})
    strata = defaultdict(lambda: {"gold_events": 0, "exact_matches_at_threshold": 0})
    for image_id, image in complete.items():
        prediction = pred_images.get(image_id, empty_prediction(image_id))
        entities = {entity["id"]: entity for entity in image["entities"]}
        relations = {relation["id"]: relation for relation in image["relations"]}
        pred_relations = {relation["id"]: relation for relation in prediction["relations"]}
        mapping = instance_correspondence(image, prediction, iou_threshold)
        ref_local, pred_local = defaultdict(list), defaultdict(list)
        for event in image.get("events", []):
            edges = event_edges(event, relations)
            row = (image_id, event, edges, structural_stratum(edges, entities))
            references[event["verb"]].append(row)
            ref_local[event["verb"]].append(row)
            counts[event["verb"]]["gold"] += len(event["relation_ids"])
            strata[row[3]]["gold_events"] += 1
        for event in prediction.get("events", []):
            edges = event_edges(event, pred_relations, mapping)
            row = (image_id, event, edges)
            predictions[event["verb"]].append(row)
            if event["score"] >= event_threshold:
                pred_local[event["verb"]].append(row)
                counts[event["verb"]]["predicted"] += len(event["relation_ids"])
        for verb in set(ref_local) | set(pred_local):
            reference_rows, prediction_rows = ref_local[verb], pred_local[verb]
            if not reference_rows or not prediction_rows:
                continue
            weights = np.asarray([[len(predicted[2] & reference[2]) for reference in reference_rows]
                                  for predicted in prediction_rows], dtype=int)
            pred_indices, ref_indices = linear_sum_assignment(-weights)
            counts[verb]["correct"] += int(weights[pred_indices, ref_indices].sum())
    per_verb = []
    for verb, reference_rows in sorted(references.items()):
        candidates = defaultdict(list)
        for image_id, event, edges, stratum in reference_rows:
            candidates[image_id].append((event, edges, stratum))
        covered = set()
        outcomes = []
        for image_id, event, edges in sorted(predictions[verb], key=lambda row: (-row[1]["score"], row[0], row[1]["id"])):
            matched = None
            if len(edges) == len(event["relation_ids"]):
                for ref_event, ref_edges, stratum in sorted(candidates[image_id], key=lambda row: row[0]["id"]):
                    if edges == ref_edges and (image_id, ref_event["id"]) not in covered:
                        matched = (image_id, ref_event["id"])
                        if event["score"] >= event_threshold:
                            strata[stratum]["exact_matches_at_threshold"] += 1
                        break
            outcomes.append((event["score"], matched is not None))
            if matched is not None:
                covered.add(matched)
        statistics = counts[verb]
        denominator = statistics["predicted"] + statistics["gold"]
        per_verb.append({"verb": verb, "gt_events": len(reference_rows),
                         "prediction_events": len(predictions[verb]),
                         "ap": average_precision(outcomes, len(reference_rows)),
                         "aligned_F1": 2 * statistics["correct"] / denominator,
                         **statistics})
    total_correct = sum(count["correct"] for count in counts.values())
    total_predicted = sum(count["predicted"] for count in counts.values())
    total_gold = sum(count["gold"] for count in counts.values())
    denominator = total_predicted + total_gold
    for statistics in strata.values():
        statistics["exact_recall_at_threshold"] = statistics["exact_matches_at_threshold"] / statistics["gold_events"]
    return {
        "status": "research_protocol_not_official_benchmark", "evaluated_images": len(complete),
        "excluded_images": len(gt_images) - len(complete),
        "mAP": mean_or_none(record["ap"] for record in per_verb),
        "aligned_macro_F1": mean_or_none(record["aligned_F1"] for record in per_verb),
        "aligned_micro_F1": 2 * total_correct / denominator if denominator else None,
        "event_threshold": event_threshold,
        "unsupported_verb_prediction_count": sum(len(rows) for verb, rows in predictions.items() if verb not in references),
        "strata": dict(strata), "verbs": per_verb,
    }


def evaluate(ground_truth, predictions, iou_threshold=0.5, event_threshold=0.5):
    require(finite_number(iou_threshold) and 0 < iou_threshold <= 1, "invalid IoU threshold")
    require(finite_number(event_threshold) and 0 <= event_threshold <= 1, "invalid event threshold")
    gt_images, pred_images, classes = validate_inputs(ground_truth, predictions)
    return {
        "protocol": VERSION, "status": "research_reference_implementation_not_release_certified",
        "annotation_policy_id": ground_truth["annotation_policy_id"],
        "iou_threshold": iou_threshold, "ap_interpolation": "all_points_precision_envelope_score_ties_grouped",
        "coordinate_convention": "continuous_xyxy_original_pixels",
        "images": len(gt_images), "missing_prediction_images": len(set(gt_images) - set(pred_images)),
        "relation": relation_ap(gt_images, pred_images, classes, iou_threshold),
        "event": event_metrics(gt_images, pred_images, iou_threshold, event_threshold),
    }
