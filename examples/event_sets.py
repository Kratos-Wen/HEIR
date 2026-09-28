"""Decode synthetic event potentials and score a complete participant-role set."""
import argparse
import copy
import json
import math

import numpy as np

from evaluation.heir.core import event_metrics
from evaluation.heir.schema import ROLES, validate_image
from evaluation.heir_sets import ground_truth, prediction
from evaluation.native import decode_state_winners, log_partition_event


def run_example():
    roles = ("target", "instrument")
    weights = np.array([[-5., 0., -8.], [-5., -8., 0.]])
    energy = np.zeros((3, 3 ** len(roles)))
    log_z = log_partition_event(weights, energy)
    hypotheses = decode_state_winners(weights, energy, log_z)
    log_probability, members = hypotheses[0]

    row = {
        "image_id": "synthetic_event", "width": 160, "height": 100,
        "boxes": [
            {"id": 0, "category": "person", "bbox": [0, 0, 50, 100]},
            {"id": 1, "category": "bread", "bbox": [70, 40, 120, 80]},
            {"id": 2, "category": "knife", "bbox": [55, 25, 100, 35]},
        ],
        "relations": [
            {"subject": 0, "object": 1, "verb": "cut", "role": "target"},
            {"subject": 0, "object": 2, "verb": "cut", "role": "instrument"},
        ],
    }
    record = {
        "image_id": row["image_id"],
        "entities": [{"id": box["id"], "noun": box["category"],
                      "box": box["bbox"], "score": 1.} for box in row["boxes"]],
        "sets": [{"subject_id": 0, "action": "cut", "score": math.exp(log_probability),
                  "members": [{"entity_id": candidate + 1, "role": roles[role]}
                              for candidate, role in members]}],
    }
    taxonomy = {"nouns": ["person", "bread", "knife"], "verbs": ["cut"], "roles": list(ROLES)}
    reference = ground_truth(row)
    validate_image(reference, taxonomy)

    def measure(item):
        predicted = prediction(item, row)
        validate_image(predicted, taxonomy, prediction=True, dimensions=(160, 100))
        return 100 * event_metrics({row["image_id"]: reference},
                                   {row["image_id"]: predicted}, .5, .5)["mAP"]

    incomplete = copy.deepcopy(record)
    incomplete["sets"][0]["members"].pop()
    return {"example": "synthetic_two_participant_event",
            "best_set": record["sets"][0], "hypotheses": len(hypotheses),
            "complete_set_map_percent": measure(record),
            "incomplete_set_map_percent": measure(incomplete)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
