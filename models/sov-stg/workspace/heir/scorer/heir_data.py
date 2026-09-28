"""HEIR v1.0 vocabulary and split loading for the scorers.

HEIR relations are (person box -> verb -> entity box, role) with six functional roles. Nothing here reads a split
other than the one requested.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


HEIR_ROLE_NAMES = ("target", "instrument", "support", "source", "destination", "constraint")
HEIR_SPLITS = ("train", "val", "test")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HEIRVocabulary:
    """Verbs in vocabulary order; nouns in detector order (person first); six roles."""

    def __init__(self, root: str | Path):
        root = Path(root)
        payload = json.loads((root / "vocabulary.json").read_text())
        self.root = root
        self.version = str(payload.get("version"))
        self.sha256 = sha256(root / "vocabulary.json")
        self.verbs = tuple(str(v["id"]) for v in payload["verbs"])
        nouns = [str(n["id"]) for n in payload["nouns"]]
        if "person" not in nouns:
            raise ValueError("HEIR vocabulary has no person noun.")
        self.objects = ("person",) + tuple(n for n in nouns if n != "person")
        self.l1 = {str(n["id"]): str(n["l1"]) for n in payload["nouns"]}
        roles = tuple(str(r["id"]) for r in payload["roles"])
        if set(roles) != set(HEIR_ROLE_NAMES):
            raise ValueError(f"HEIR roles {roles} differ from the six-role protocol {HEIR_ROLE_NAMES}.")
        self.role_names = HEIR_ROLE_NAMES
        if len(set(self.verbs)) != len(self.verbs) or len(set(self.objects)) != len(self.objects):
            raise ValueError("Duplicate verb or noun ids in the vocabulary.")
        self.verb_index = {name: i for i, name in enumerate(self.verbs)}
        self.object_index = {name: i for i, name in enumerate(self.objects)}
        self.role_index = {name: i for i, name in enumerate(self.role_names)}


@dataclass(frozen=True)
class HEIRImage:
    image_id: str
    file_name: str
    width: int
    height: int
    source: str
    evaluation: str | None
    boxes: tuple[tuple[float, float, float, float], ...]
    categories: tuple[int, ...]
    # (subject box index, object box index, verb index, role index); subjects are persons.
    relations: tuple[tuple[int, int, int, int], ...]


def load_heir_split(root: str | Path, split: str, vocab: HEIRVocabulary) -> tuple[HEIRImage, ...]:
    if split not in HEIR_SPLITS:
        raise ValueError(f"Unknown HEIR split {split!r}.")
    path = Path(root) / "annotations" / f"{split}.json"
    data = json.loads(path.read_text())
    if data.get("split") != split:
        raise ValueError(f"{path} declares split {data.get('split')!r}.")
    images, seen = [], set()
    for record in data["images"]:
        image_id = str(record["image_id"])
        if image_id in seen:
            raise ValueError(f"Duplicate image {image_id}.")
        seen.add(image_id)
        width, height = int(record["width"]), int(record["height"])
        ids = [int(b["id"]) for b in record["boxes"]]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"{image_id}: box ids must be contiguous from 1.")
        boxes, categories = [], []
        for box in record["boxes"]:
            x1, y1, x2, y2 = (float(v) for v in box["bbox"])
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"{image_id}: invalid box {box['bbox']}.")
            boxes.append((x1, y1, x2, y2))
            categories.append(vocab.object_index[str(box["category"])])
        relations = set()
        for relation in record["relations"]:
            subject, entity = int(relation["subject"]) - 1, int(relation["object"]) - 1
            if not (0 <= subject < len(boxes) and 0 <= entity < len(boxes)) or subject == entity:
                raise ValueError(f"{image_id}: relation endpoints invalid {relation}.")
            if categories[subject] != 0:
                raise ValueError(f"{image_id}: relation subject is not a person.")
            relations.add((subject, entity, vocab.verb_index[str(relation["verb"])],
                           vocab.role_index[str(relation["role"])]))
        evaluation = record.get("evaluation")
        if evaluation not in (None, "agent_only"):
            raise ValueError(f"{image_id}: unknown evaluation flag {evaluation!r}.")
        images.append(HEIRImage(image_id, str(record["file_name"]), width, height, str(record.get("source", "")),
                                evaluation, tuple(boxes), tuple(categories), tuple(sorted(relations))))
    return tuple(images)
