#!/usr/bin/env python3
"""Evaluate complete person--action sets from HEIR baseline predictions.

Pair-output predictions are grouped into identities: same-noun boxes join fixed representatives at IoU 0.7 and
duplicate edges keep their maximum score. Two decoders form set hypotheses for each person--action pair:
top-k decoding (the k highest-scoring edges for every k, scored by the minimum member confidence) and MAP decoding
(one best assignment per (cardinality, role-count) state of the independent edge product, scored by P(S)). At most
100 sets per image are retained; there is no additional relation cap by default. Ground truth groups annotated
entity--role members by person and action. Scoring uses shared, noun-compatible image-level entity matching at
IoU 0.5 and exact complete-set matches, averaging AP over the actions supported by the evaluated split.
"""
import argparse, json, os, subprocess, sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from heir_eval_convert import build_ground_truth, build_predictions, iou, sha256  # noqa: E402
from state_map import MAX_EVENT_SETS, decode_state_winners, log_partition_event  # noqa: E402
from heir_data import HEIRVocabulary, load_heir_split  # noqa: E402

ROLES = ("target", "instrument", "support", "source", "destination", "constraint")


def load_records(path, inter, max_relations):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        nodes = [{"id": i, "box": b, "category_id": int(l), "detector_score": float(s)}
                 for i, (b, l, s) in enumerate(zip(p["boxes"], p["labels"], p["box_scores"]))]
        hois = sorted(p["hois"], key=lambda h: -h["score"])[:max_relations]
        rels = [{"subject": int(h["subject_id"]), "entity": int(h["object_id"]), "verb": inter[h["category_id"]][0],
                 "role": inter[h["category_id"]][1], "score": float(h["score"])} for h in hois
                if int(h["subject_id"]) != int(h["object_id"])]
        records.append({"image_id": p["image_id"], "nodes": nodes, "relations": rels, "events": [], "sets": p.get("sets", [])})
    return records


def native_sets(record, img, vocab, stats):
    """Native set interface: the model's own scored set hypotheses (see SET_MAP_PROTOCOL.md).

    JSONL "sets": [{"subject_id": box index, "verb": verb name, "members": [{"object_id": box index, "role": role name}, ...],
    "score": s in [0, 1]}]. Boxes are the model's entities (no clustering); the agent_only rule is applied at the set level
    (subject must match an annotated agent at IoU >= 0.5 and the verb must be that agent's); degenerate boxes, self members,
    repeated members, unknown verbs/roles and sets without members are rejected as invalid input.
    """
    width, height = float(img.width), float(img.height)
    entities, valid = [], {}
    for node in record["nodes"]:
        x1, y1, x2, y2 = node["box"]
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if not (x1 < x2 and y1 < y2):
            stats["degenerate_nodes"] += 1
            continue
        valid[node["id"]] = f"node_{node['id']}"
        entities.append({"id": valid[node["id"]], "noun": vocab.objects[int(node["category_id"])],
                         "bbox": [x1, y1, x2, y2], "score": float(min(1.0, max(0.0, node["detector_score"])))})
    boxes = {e["id"]: e["bbox"] for e in entities}
    agent_verbs = None
    if img.evaluation == "agent_only":
        agent_verbs = {}
        for s, _, v, _ in img.relations:
            agent_verbs.setdefault(s, set()).add(vocab.verbs[v])
    edges, events = {}, []
    for hyp in record["sets"]:
        verb = hyp["verb"]
        if verb not in vocab.verbs:
            raise ValueError(f"{record['image_id']}: unknown verb {verb!r}")
        if hyp["subject_id"] not in valid:
            stats["sets_with_invalid_subject"] += 1
            continue
        sid = valid[hyp["subject_id"]]
        if agent_verbs is not None and not any(iou(boxes[sid], img.boxes[a]) >= 0.5 and verb in verbs
                                               for a, verbs in agent_verbs.items()):
            stats["agent_only_ignored_sets"] += 1
            continue
        members = []
        for m in hyp["members"]:
            if m["role"] not in ROLES:
                raise ValueError(f"{record['image_id']}: unknown role {m['role']!r}")
            if m["object_id"] not in valid or valid[m["object_id"]] == sid:
                stats["sets_with_invalid_member"] += 1
                members = None
                break
            members.append((sid, valid[m["object_id"]], verb, m["role"]))
        if not members:
            stats["empty_or_invalid_sets"] += 1
            continue
        if len(set(members)) != len(members):
            raise ValueError(f"{record['image_id']}: repeated member inside a set")
        score = float(hyp["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{record['image_id']}: set score outside [0, 1]")
        for key in members:
            edges[key] = max(edges.get(key, 0.0), score)
        events.append((members, score))
    relations, rel_id = [], {}
    for key, score in sorted(edges.items()):
        rel_id[key] = f"rel_{len(relations)}"
        relations.append({"id": rel_id[key], "subject_id": key[0], "entity_id": key[1], "verb": key[2], "role": key[3], "score": score})
    out = [{"verb": ms[0][2], "relation_ids": sorted(rel_id[k] for k in ms), "score": sc} for ms, sc in events]
    out.sort(key=lambda e: (-e["score"], e["verb"], e["relation_ids"]))
    return entities, relations, out


def canonical_events(gt):
    n = 0
    for image in gt["images"]:
        groups = defaultdict(list)
        for r in image["relations"]:
            groups[(r["subject_id"], r["verb"])].append(r["id"])
        image["events"] = [{"id": f"set_{k}", "verb": verb, "relation_ids": sorted(members)}
                           for k, ((_, verb), members) in enumerate(sorted(groups.items()))]
        image["events_complete"] = True
        n += len(image["events"])
    return n


def cluster_identities(image, identity_iou, stats):
    """Return (entities, edges) at identity level; edges = {(sid, eid, verb, role): score}."""
    ent = {e["id"]: e for e in image["entities"]}
    priority = defaultdict(lambda: -1.0)
    for r in image["relations"]:
        for end in (r["subject_id"], r["entity_id"]):
            priority[end] = max(priority[end], r["score"])
    order = sorted(priority, key=lambda i: (-priority[i], -ent[i]["score"], i))
    identities, assign = [], {}
    for i in order:
        e = ent[i]
        best, best_k = -1.0, None
        for k, ident in enumerate(identities):
            if ident["noun"] != e["noun"]:
                continue
            o = iou(e["bbox"], ident["bbox"])
            if o >= identity_iou and o > best:
                best, best_k = o, k
        if best_k is None:
            best_k = len(identities)
            identities.append({"id": f"ident_{best_k}", "noun": e["noun"], "bbox": e["bbox"], "score": e["score"]})
        assign[i] = identities[best_k]["id"]
    edges = {}
    for r in image["relations"]:
        s, o = assign[r["subject_id"]], assign[r["entity_id"]]
        if s == o:
            stats["self_edges_after_clustering"] += 1
            continue
        key = (s, o, r["verb"], r["role"])
        edges[key] = max(edges.get(key, 0.0), r["score"])
    stats["boxes"] += len(order)
    stats["identities"] += len(identities)
    return identities, edges


def map_event(members):
    """Per-state MAP hypotheses of one (person identity, action) event under the independent product of its edges.

    members: [(relation id, entity id, role, score)]. Each candidate entity takes null or one role; its log weights are
    log p for each role and log(1 - sum_r p) for null (probabilities are rescaled when their sum reaches 1), and the
    count potentials are zero. The event is decoded with the same function as the native CoRISP sets: the best assignment
    of every nonempty (cardinality, role-count) state, at most eight, scored by its normalised probability P(S).
    """
    entities = sorted({e for _, e, _, _ in members})
    roles = [r for r in ROLES if any(m[2] == r for m in members)]
    prob = np.zeros((len(entities), len(roles)))
    rid = {}
    for rel, e, r, score in members:
        i, j = entities.index(e), roles.index(r)
        prob[i, j] = max(prob[i, j], score)
        rid[i, j] = rel
    total = prob.sum(1, keepdims=True)
    prob = np.where(total > 1 - 1e-4, prob * (1 - 1e-4) / np.maximum(total, 1e-12), prob)
    with np.errstate(divide="ignore"):
        weights = np.concatenate([np.log1p(-prob.sum(1, keepdims=True)), np.log(prob)], 1)
    energy = np.zeros((len(entities) + 1, 3 ** len(roles)))
    log_z = log_partition_event(weights, energy)
    out = []
    for log_p, chosen in decode_state_winners(weights, energy, log_z, MAX_EVENT_SETS):
        out.append((sorted(rid[i, j] for i, j in chosen), float(np.exp(min(0.0, log_p)))))
    return out


def build_sets(identities, edges, tau, set_score, max_sets, stats, construction="threshold"):
    relations, rel_id = [], {}
    for key, score in sorted(edges.items()):
        rel_id[key] = f"rel_{len(relations)}"
        relations.append({"id": rel_id[key], "subject_id": key[0], "entity_id": key[1], "verb": key[2],
                          "role": key[3], "score": score})
    groups = defaultdict(list)
    for key, score in edges.items():
        if construction in ("top-k", "map") or score >= tau:
            groups[(key[0], key[2])].append((rel_id[key], score))
    events = []
    for (sid, verb), members in sorted(groups.items()):
        if construction == "map":
            by_id = {r["id"]: r for r in relations}
            for rel_ids, score in map_event([(m, by_id[m]["entity_id"], by_id[m]["role"], sc) for m, sc in members]):
                events.append({"verb": verb, "relation_ids": rel_ids, "score": score})
            continue
        if construction == "top-k":
            # threshold-free: nested hypotheses top-1, top-2, ... members by score; each scored by its weakest member
            members = sorted(members, key=lambda m: (-m[1], m[0]))
            for k in range(1, len(members) + 1):
                scores = [s for _, s in members[:k]]
                events.append({"verb": verb, "relation_ids": sorted(m for m, _ in members[:k]),
                               "score": float(min(scores) if set_score == "min" else sum(scores) / len(scores))})
            continue
        scores = [s for _, s in members]
        events.append({"verb": verb, "relation_ids": sorted(m for m, _ in members),
                       "score": float(min(scores) if set_score == "min" else sum(scores) / len(scores))})
    events.sort(key=lambda e: (-e["score"], e["verb"], e["relation_ids"]))
    if len(events) > max_sets:
        stats["sets_dropped_by_cap"] += len(events) - max_sets
        events = events[:max_sets]
    for k, e in enumerate(events):
        e["id"] = f"set_{k}"
    stats["sets"] += len(events)
    stats["set_members"] += sum(len(e["relation_ids"]) for e in events)
    return relations, events


def run_heir_eval(gt_path, pred_path):
    env = dict(os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = subprocess.run([sys.executable, "-m", "heir_eval_v02", "detection", "--gt", str(gt_path), "--pred", str(pred_path)],
                         capture_output=True, text=True, env=env)
    if run.returncode:
        raise RuntimeError("heir_eval refused the inputs:\n" + run.stderr[-3000:])
    return json.loads(run.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heir-root", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-relations-per-image", type=int, default=None,
                    help="Optional relation cap; by default only the set budget is applied")
    ap.add_argument("--identity-iou", type=float, default=0.7)
    ap.add_argument("--set-score", choices=("min", "mean"), default="min")
    ap.add_argument("--max-sets", type=int, default=100)
    ap.add_argument("--construction", choices=("top-k", "map", "native", "threshold"), default="top-k",
                    help="top-k (default): for each person--action pair, the k highest-scoring edges for every k; "
                         "map: MAP decoding of the independent product of the edges, one best assignment per (cardinality, role-count) state (the CoRISP decoder); "
                         "native: the model's own scored sets from the JSONL 'sets' field; threshold: members = edges >= tau")
    ap.add_argument("--participants", choices=("all", "single", "multi", "repeated", "shared_strict"), default="all",
                    help="structural stratum: sets with exactly one / at least two participants / a role filled by >= 2 members (both GT and hypotheses)")
    ap.add_argument("--image-subset", choices=("all", "shared_entity"), default="all",
                    help="shared_entity: images whose GT has an entity participating in events of >= 2 different persons")
    ap.add_argument("--tau", type=float, help="fixed member threshold (test)")
    ap.add_argument("--tau-grid", type=str, help="comma-separated thresholds; the best on this split is selected (val)")
    a = ap.parse_args()
    if a.construction in ("top-k", "map", "native"):
        a.tau, a.tau_grid = 0.0, None
    elif (a.tau is None) == (a.tau_grid is None):
        raise SystemExit("give exactly one of --tau / --tau-grid")
    classes = json.loads(a.classes.read_text())
    vocab = HEIRVocabulary(a.heir_root)
    if list(vocab.objects) != classes["objects"]:
        raise ValueError("Object order differs between classes.json and the HEIR vocabulary.")
    train = load_heir_split(a.heir_root, "train", vocab)
    split = load_heir_split(a.heir_root, a.split, vocab)
    records = load_records(a.predictions, classes["interaction_verb_role"], a.max_relations_per_image)
    if [r["image_id"] for r in records] != [i.image_id for i in split]:
        raise ValueError("Predictions must cover the declared split in order (empty images included).")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    gt = build_ground_truth(a.heir_root, a.split, vocab, train, split)
    gt_sets = canonical_events(gt)
    def keep_event(members):
        """members: list of (entity_id, role). Strata: single / multi (participant count), repeated (a role filled by >= 2 members)."""
        if a.participants == "single":
            return len(members) == 1
        if a.participants == "multi":
            return len(members) >= 2
        if a.participants == "repeated":
            roles = [r for _, r in members]
            return len(roles) != len(set(roles))
        return True
    if a.image_subset == "shared_entity":
        # images whose ground truth has one entity participating in events of >= 2 different persons
        kept_images = set()
        for image in gt["images"]:
            owners = defaultdict(set)
            for r in image["relations"]:
                owners[r["entity_id"]].add(r["subject_id"])
            if any(len(v) >= 2 for v in owners.values()):
                kept_images.add(image["id"])
        gt["images"] = [im for im in gt["images"] if im["id"] in kept_images]
        records = [r for r in records if r["image_id"] in kept_images]
        split = [im for im in split if im.image_id in kept_images]
        gt_sets = sum(len(im["events"]) for im in gt["images"])
        present = {(r["verb"], next(e["noun"] for e in im["entities"] if e["id"] == r["entity_id"]), r["role"]) for im in gt["images"] for r in im["relations"]}
        gt["classes"] = [c for c in gt["classes"] if (c["verb"], c["noun"], c["role"]) in present]
    if a.participants == "shared_strict":
        # Strict shared-entity stratum: GROUND-TRUTH sets containing an entity that also participates in a set of a
        # different person; hypotheses are NOT filtered (no GT access on the prediction side), so all other hypotheses of
        # the image count as false positives: a conservative, method-consistent lower bound.
        for image in gt["images"]:
            rel = {r["id"]: r for r in image["relations"]}
            owners = defaultdict(set)
            for r in image["relations"]:
                owners[r["entity_id"]].add(r["subject_id"])
            image["events"] = [e for e in image["events"] if any(len(owners[rel[x]["entity_id"]]) >= 2 for x in e["relation_ids"])]
            kept = {r for e in image["events"] for r in e["relation_ids"]}
            image["relations"] = [r for r in image["relations"] if r["id"] in kept]
    elif a.participants != "all":
        # Structural stratum applied to BOTH ground-truth sets and hypotheses. An exact match requires identical member
        # multisets, so the strata cannot interact and the restricted AP needs no ignore rule.
        for image in gt["images"]:
            rel = {r["id"]: r for r in image["relations"]}
            image["events"] = [e for e in image["events"] if keep_event([(rel[x]["entity_id"], rel[x]["role"]) for x in e["relation_ids"]])]
            kept = {r for e in image["events"] for r in e["relation_ids"]}
            image["relations"] = [r for r in image["relations"] if r["id"] in kept]
    if a.participants != "all":
        gt_sets = sum(len(im["events"]) for im in gt["images"])
        present = {(r["verb"], next(e["noun"] for e in im["entities"] if e["id"] == r["entity_id"]), r["role"]) for im in gt["images"] for r in im["relations"]}
        gt["classes"] = [c for c in gt["classes"] if (c["verb"], c["noun"], c["role"]) in present]
    gt_path = a.output_dir / "gt_sets.json"
    gt_path.write_text(json.dumps(gt))
    stats = Counter()
    schema = "heir-eval-research-0.2"
    if a.construction == "native":
        by_id = {img.image_id: img for img in split}
        native = []
        for record in records:
            entities, relations, events = native_sets(record, by_id[record["image_id"]], vocab, stats)
            if len(events) > a.max_sets:
                stats["sets_dropped_by_cap"] += len(events) - a.max_sets
                events = events[:a.max_sets]
            for k, e in enumerate(events):
                e["id"] = f"set_{k}"
            stats["sets"] += len(events)
            stats["set_members"] += sum(len(e["relation_ids"]) for e in events)
            native.append({"id": record["image_id"], "entities": entities, "relations": relations, "events": events})
        conv = {"images": len(native), "native_sets_submitted": sum(len(r["sets"]) for r in records)}
    else:
        pred, conv = build_predictions(records, vocab, split)  # agent_only pre-filter applied here
        schema = pred["schema_version"]
        clustered = [(img["id"], *cluster_identities(img, a.identity_iou, stats)) for img in pred["images"]]

    def score(tau, tag):
        st = Counter()
        if a.construction == "native":
            images = native
        else:
            images = []
            for image_id, identities, edges in clustered:
                relations, events = build_sets(identities, edges, tau, a.set_score, a.max_sets, st, a.construction)
                images.append({"id": image_id, "entities": identities, "relations": relations, "events": events})
        if a.participants not in ("all", "shared_strict"):
            for image in images:
                rel = {r["id"]: r for r in image["relations"]}
                image["events"] = [e for e in image["events"] if keep_event([(rel[x]["entity_id"], rel[x]["role"]) for x in e["relation_ids"]])]
        pred_path = a.output_dir / f"pred_sets_{tag}.json"
        pred_path.write_text(json.dumps({"schema_version": schema, "images": images}))
        result = run_heir_eval(gt_path, pred_path)
        return result, dict(st), pred_path

    grid = {}
    if a.tau_grid:
        for tau in [float(x) for x in a.tau_grid.split(",")]:
            result, st, pred_path = score(tau, f"tau{tau:g}")
            grid[tau] = {"set_mAP": result["result"]["event"]["mAP"], "sets": st["sets"], "members": st["set_members"]}
            print(f"tau={tau:g} set_mAP={grid[tau]['set_mAP']:.4f} sets={st['sets']}", flush=True)
            pred_path.unlink()
        best = max(sorted(grid), key=lambda t: (grid[t]["set_mAP"], -t))
        tau = best
    else:
        tau = a.tau
    result, st, pred_path = score(tau, "final")
    ev = result["result"]["event"]
    summary = {"split": a.split, "metric": "set_mAP", "images": len(split), "set_mAP": ev["mAP"],
               "verbs_with_gt_sets": len(ev["verbs"]), "gt_sets": gt_sets, "tau": tau,
               "tau_selection": ("selected on this split by set mAP" if a.tau_grid else "fixed (from val)"),
               "tau_grid": {f"{t:g}": v for t, v in grid.items()},
               "construction": a.construction, "participants": a.participants, "image_subset": a.image_subset, "identity_iou": a.identity_iou, "set_score": a.set_score, "max_sets_per_image": a.max_sets,
               "prediction_stats": {**conv, **dict(stats), **st},
               "predictions_sha256": sha256(a.predictions), "evaluator": "heir_eval research 0.2 (event metrics)",
               "heir_annotation_sha256": {s_: sha256(a.heir_root / "annotations" / f"{s_}.json") for s_ in ("train", a.split)}}
    (a.output_dir / "heir_set_metrics.json").write_text(json.dumps(result, indent=1))
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("split", "set_mAP", "tau", "verbs_with_gt_sets", "gt_sets")}, indent=1))


if __name__ == "__main__":
    main()
