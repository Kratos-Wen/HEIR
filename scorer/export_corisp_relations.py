#!/usr/bin/env python3
"""Convert CoRISP predictions to the relation format of evaluate_heir_predictions.py.

Input: the JSONL written by corisp_heir.predict on the main branch, with `entities`, `pairs` and dense `role_scores`
indexed by pair, action and role. Output: one line per image of the split, in annotation order, holding the highest
`role_scores` as `hois`, and a classes JSON with all 105 x 6 (verb, role) interactions, where an interaction id is
verb_index * 6 + role_index.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heir_data import HEIR_ROLE_NAMES, HEIRVocabulary, load_heir_split  # noqa: E402


def interactions(vocab):
    return [[verb, role] for verb in vocab.verbs for role in HEIR_ROLE_NAMES]


def convert(record, vocab, max_relations):
    entities = record["entities"]
    pairs, actions, roles = record["pairs"], record["actions"], record["roles"]
    scores = np.asarray(record["role_scores"], dtype=np.float64).reshape(len(pairs), len(actions), len(roles))
    position = {entity["id"]: i for i, entity in enumerate(entities)}
    hois = []
    for index in np.argsort(-scores.ravel(), kind="stable")[:max_relations]:
        p, a, r = np.unravel_index(index, scores.shape)
        category = vocab.verb_index[actions[a]] * len(HEIR_ROLE_NAMES) + vocab.role_index[roles[r]]
        hois.append({"subject_id": position[pairs[p][0]], "object_id": position[pairs[p][1]], "category_id": category,
                     "score": min(1.0, max(0.0, float(scores[p, a, r])))})
    return {"image_id": record["image_id"], "boxes": [[round(x, 2) for x in e["box"]] for e in entities],
            "labels": [vocab.object_index[e["noun"]] for e in entities],
            "box_scores": [round(min(1.0, max(0.0, e["score"])), 6) for e in entities], "hois": hois}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heir-root", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="JSONL from corisp_heir.predict")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--classes-out", type=Path, required=True)
    ap.add_argument("--max-relations-per-image", type=int, default=100)
    a = ap.parse_args()
    vocab = HEIRVocabulary(a.heir_root)
    converted = {}
    with a.predictions.open() as handle:
        for line in handle:
            if line.strip():
                record = convert(json.loads(line), vocab, a.max_relations_per_image)
                converted[record["image_id"]] = record
    order = [image.image_id for image in load_heir_split(a.heir_root, a.split, vocab)]
    if set(order) != set(converted):
        raise ValueError("Predictions must cover exactly the declared split.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as handle:
        for image_id in order:
            handle.write(json.dumps(converted[image_id]) + "\n")
    names = interactions(vocab)
    a.classes_out.parent.mkdir(parents=True, exist_ok=True)
    a.classes_out.write_text(json.dumps({"objects": list(vocab.objects), "interaction_verb_role": names,
                                         "interactions": [f"{verb} {role}" for verb, role in names]}, indent=1))
    print(f"{a.output}: {len(order)} images, {sum(len(r['hois']) for r in converted.values())} relations")


if __name__ == "__main__":
    main()
