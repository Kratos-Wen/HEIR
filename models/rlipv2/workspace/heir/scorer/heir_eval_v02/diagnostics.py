import hashlib
import json
from collections import defaultdict

from .core import average_precision, mean_or_none
from .schema import ROLES, identifier, probability, require, unique_records


def validate_candidates(candidates):
    unique_records(candidates, "diagnostic candidates")
    for candidate in candidates:
        require(all(identifier(candidate.get(field)) for field in ("verb", "query_id", "entity_id")),
                "diagnostic candidate missing query, entity or verb")
        require(candidate.get("role") in ROLES, "invalid diagnostic role")
        require(isinstance(candidate.get("positive"), bool) and candidate.get("verified") is True,
                "diagnostic labels require explicit verified positive/negative decisions")
        require(probability(candidate.get("score")), "invalid diagnostic score")
    semantic_keys = [(entry["query_id"], entry["entity_id"], entry["verb"], entry["role"]) for entry in candidates]
    require(len(semantic_keys) == len(set(semantic_keys)), "duplicate diagnostic assertion")


def candidate_fingerprint(candidates):
    payload = json.dumps(sorted(candidates, key=lambda entry: entry["id"]), sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def binding_ap(candidates):
    validate_candidates(candidates)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["verb"], candidate["role"])].append(candidate)
    rows = []
    for (verb, role), group in sorted(groups.items()):
        positives = sum(candidate["positive"] for candidate in group)
        rows.append({"verb": verb, "role": role, "positives": positives, "candidates": len(group),
                     "ap": average_precision([(entry["score"], entry["positive"]) for entry in group], positives)})
    per_role = {role: mean_or_none(row["ap"] for row in rows if row["role"] == role and row["ap"] is not None)
                for role in ROLES}
    return {"balanced_role_mAP": mean_or_none(value for value in per_role.values() if value is not None),
            "per_role": per_role, "groups": rows,
            "unsupported_negative_candidates": sum(row["candidates"] for row in rows if row["positives"] == 0),
            "candidate_sha256": candidate_fingerprint(candidates),
            "setting": "oracle_person_verb_and_candidate_boxes_not_end_to_end"}


def calibrate_operating_point(candidates, validation_split_id, target_recall=0.9):
    validate_candidates(candidates)
    require(identifier(validation_split_id), "missing validation split id")
    require(probability(target_recall) and target_recall > 0, "invalid target recall")
    positives = [candidate["score"] for candidate in candidates if candidate["positive"]]
    require(positives, "cannot calibrate without positive validation candidates")
    ranked_positives = sorted(positives, reverse=True)
    threshold = next(value for rank, value in enumerate(ranked_positives, start=1)
                     if rank / len(positives) >= target_recall)
    return {"validation_split_id": validation_split_id, "threshold": threshold,
            "target_validation_recall": target_recall,
            "achieved_validation_recall": sum(score >= threshold for score in positives) / len(positives),
            "validation_sha256": candidate_fingerprint(candidates)}


def negative_report(candidates, calibration, test_split_id):
    validate_candidates(candidates)
    require(isinstance(calibration, dict), "calibration must be an object")
    digest = calibration.get("validation_sha256")
    require(isinstance(digest, str) and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "missing or invalid calibration fingerprint")
    require(identifier(test_split_id) and test_split_id != calibration.get("validation_split_id"),
            "test split must differ from calibration split")
    require(identifier(calibration.get("validation_split_id")) and probability(calibration.get("threshold")),
            "invalid frozen calibration artifact")
    require(calibration.get("validation_sha256") != candidate_fingerprint(candidates),
            "test candidates are identical to calibration candidates")
    threshold = calibration["threshold"]
    positives = [candidate for candidate in candidates if candidate["positive"]]
    negatives = [candidate for candidate in candidates if not candidate["positive"]]
    return {
        "threshold": threshold, "validation_split_id": calibration["validation_split_id"],
        "test_split_id": test_split_id, "positives": len(positives), "negatives": len(negatives),
        "actual_test_recall": mean_or_none(candidate["score"] >= threshold for candidate in positives),
        "hard_negative_FPR": mean_or_none(candidate["score"] >= threshold for candidate in negatives),
        "operating_point": "validation_calibrated_not_test_FPR_at_fixed_TPR",
        "test_sha256": candidate_fingerprint(candidates),
    }
