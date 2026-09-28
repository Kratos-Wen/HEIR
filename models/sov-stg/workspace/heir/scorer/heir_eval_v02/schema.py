import math


VERSION = "heir-eval-research-0.2"
ROLES = (
    "target", "instrument", "support",
    "source", "destination", "constraint",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value):
    return isinstance(value, str) and bool(value.strip())


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def probability(value):
    return finite_number(value) and 0 <= value <= 1


def unique_records(records, context):
    require(isinstance(records, list), f"{context}: expected a list")
    result = {}
    for record in records:
        require(isinstance(record, dict), f"{context}: expected an object")
        record_id = record.get("id")
        require(identifier(record_id), f"{context}: invalid id")
        require(record_id not in result, f"{context}: duplicate id {record_id}")
        result[record_id] = record
    return result


def relation_key(relation):
    return (relation["subject_id"], relation["entity_id"], relation["verb"], relation["role"])


def class_key(relation, entities):
    return (relation["verb"], entities[relation["entity_id"]]["noun"], relation["role"])


def validate_image(image, taxonomy, prediction=False, dimensions=None):
    require(isinstance(image, dict) and identifier(image.get("id")), "invalid image id")
    context = f"image {image['id']}"
    if dimensions is None:
        width, height = image.get("width"), image.get("height")
    else:
        width, height = dimensions
    require(finite_number(width) and width > 0 and finite_number(height) and height > 0,
            f"{context}: invalid dimensions")
    entities = unique_records(image.get("entities"), context + " entities")
    relations = unique_records(image.get("relations"), context + " relations")
    events = unique_records(image.get("events", []), context + " events")
    for entity in entities.values():
        require(entity.get("noun") in taxonomy["nouns"], f"{context}: unknown noun")
        bbox = entity.get("bbox")
        require(isinstance(bbox, list) and len(bbox) == 4 and all(map(finite_number, bbox)),
                f"{context}: invalid bbox")
        require(0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height,
                f"{context}: bbox outside continuous xyxy image coordinates")
        if prediction:
            require(probability(entity.get("score")), f"{context}: invalid entity score")
    seen_relations = set()
    for relation in relations.values():
        require(identifier(relation.get("subject_id")) and identifier(relation.get("entity_id")),
                f"{context}: invalid relation endpoint id")
        require(relation.get("subject_id") in entities and relation.get("entity_id") in entities,
                f"{context}: dangling relation endpoint")
        require(relation["subject_id"] != relation["entity_id"], f"{context}: self relation")
        require(relation.get("verb") in taxonomy["verbs"], f"{context}: unknown verb")
        require(relation.get("role") in ROLES, f"{context}: unknown role")
        if prediction:
            require(probability(relation.get("score")), f"{context}: invalid relation score")
        else:
            require(entities[relation["subject_id"]]["noun"] == "person",
                    f"{context}: GT subject is not a person")
            require(relation_key(relation) not in seen_relations, f"{context}: duplicate GT relation")
        seen_relations.add(relation_key(relation))
    seen_events = set()
    referenced = set()
    for event in events.values():
        require(event.get("verb") in taxonomy["verbs"], f"{context}: unknown event verb")
        members = event.get("relation_ids")
        require(isinstance(members, list) and members and all(identifier(member) for member in members),
                f"{context}: event must contain relation ids")
        require(len(members) == len(set(members)), f"{context}: repeated event member id")
        require(all(member in relations for member in members), f"{context}: dangling event member")
        require(all(relations[member]["verb"] == event["verb"] for member in members),
                f"{context}: mixed verbs within an atomic event")
        signature = frozenset(relation_key(relations[member]) for member in members)
        require(len(signature) == len(members), f"{context}: duplicate semantic edge inside event")
        if prediction:
            require(probability(event.get("score")), f"{context}: invalid event score")
        else:
            require(signature not in seen_events, f"{context}: indistinguishable duplicate GT events")
        seen_events.add(signature)
        referenced.update(members)
    if not prediction:
        require(image.get("coverage") == "adjudicated_closed_world",
                f"{context}: no explicit complete relation coverage; refusing closed-world evaluation")
        require(isinstance(image.get("events_complete"), bool), f"{context}: missing event coverage flag")
        require(image["events_complete"] or not events,
                f"{context}: partial event GT unsupported; use a separate adjudicated subset")
        if image["events_complete"]:
            require(referenced == set(relations), f"{context}: event GT does not cover all relations")
        require(identifier(image.get("cluster_id")), f"{context}: missing source/duplicate cluster id")
    return entities, relations, events


def validate_inputs(ground_truth, predictions):
    require(isinstance(ground_truth, dict) and ground_truth.get("schema_version") == VERSION,
            "unsupported GT schema version")
    require(isinstance(predictions, dict) and predictions.get("schema_version") == VERSION,
            "unsupported prediction schema version")
    require(identifier(ground_truth.get("annotation_policy_id")), "missing annotation policy id")
    taxonomy = ground_truth.get("taxonomy", {})
    require(isinstance(taxonomy, dict), "taxonomy must be an object")
    for field in ("verbs", "nouns", "roles"):
        values = taxonomy.get(field)
        require(isinstance(values, list) and values and all(identifier(value) for value in values),
                f"invalid taxonomy {field}")
        require(len(values) == len(set(values)), f"duplicate taxonomy {field}")
    require(set(taxonomy["roles"]) == set(ROLES), "HEIR role taxonomy must contain exactly the six HEIR roles")
    require("person" in taxonomy["nouns"], "person missing from noun taxonomy")
    gt_images = unique_records(ground_truth.get("images"), "GT images")
    require(gt_images, "empty GT manifest")
    pred_images = unique_records(predictions.get("images"), "prediction images")
    require(set(pred_images) <= set(gt_images), "prediction contains unknown image ids")
    observed_classes = set()
    for image_id, image in gt_images.items():
        entities, relations, _ = validate_image(image, taxonomy)
        observed_classes.update(class_key(relation, entities) for relation in relations.values())
        if image_id in pred_images:
            validate_image(pred_images[image_id], taxonomy, True, (image["width"], image["height"]))
    declared = ground_truth.get("classes")
    require(isinstance(declared, list), "missing frozen evaluation classes")
    classes = {}
    for entry in declared:
        require(isinstance(entry, dict), "invalid evaluation class")
        key = (entry.get("verb"), entry.get("noun"), entry.get("role"))
        require(key[0] in taxonomy["verbs"] and key[1] in taxonomy["nouns"] and key[2] in ROLES,
                "invalid evaluation class labels")
        count = entry.get("train_count")
        require(isinstance(count, int) and not isinstance(count, bool) and count >= 0,
                "train_count must be a nonnegative integer from the training split")
        require(key not in classes, "duplicate evaluation class")
        classes[key] = count
    require(set(classes) == observed_classes, "frozen classes must equal GT-supported classes; no dropping classes")
    return gt_images, pred_images, classes
