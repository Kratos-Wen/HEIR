"""Official evaluation with exhaustive indexed-collector equivalence checking."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from reproductions.incom_net.backbones import WORKSPACE, digest
from reproductions.incom_net.run import DATA


def audited_evaluator(base, checked):
    class AuditedEvaluator(base):
        def __setattr__(self, name, value):
            if name == '_collect_detections_for_image' and callable(value):
                old = self._collect_detections_for_image
                # The wrapper's new collector uses its own indexed prediction cache.
                # Validate ALL outputs against the original routine, including empty images.
                for image_id in self.image_ids:
                    actual = value(None, image_id)
                    records = self._audit_records.get(int(image_id), [])
                    expected = old(records, image_id)
                    if not all(np.array_equal(x, y, equal_nan=True) for x, y in zip(actual, expected)):
                        raise ValueError(f'Official collector mismatch for image {image_id}')
                    checked['images'] += 1
                    checked['empty_images'] += int(not records)
                # Explicit empty-image behavior, even when every real image has detections.
                expected = old([], -1)
                actual = value(None, -1)
                if not all(np.array_equal(x, y, equal_nan=True) for x, y in zip(actual, expected)):
                    raise ValueError('Official empty-image collector mismatch')
                checked['empty_case_checked'] = True
            super().__setattr__(name, value)
    return AuditedEvaluator


def main():
    import pickle
    from collections import defaultdict
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output-json', type=Path, required=True)
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError('Preserve existing metrics')
    coverage = json.loads((args.cache.parent / 'coverage.json').read_text())
    ids = [int(x) for x in (DATA / 'data/splits/vcoco_test.ids').read_text().split()]
    if len(coverage['image_ids']) != len(set(coverage['image_ids'])) or set(coverage['image_ids']) != set(ids) or len(ids) != 4946:
        raise ValueError('Incomplete official test inference')
    with args.cache.open('rb') as f:
        records = pickle.load(f)
    grouped = defaultdict(list)
    allowed_ids = set(ids)
    for r in records:
        if r['image_id'] not in allowed_ids:
            raise ValueError('Out-of-test prediction')
        grouped[r['image_id']].append(r)
    source = WORKSPACE / 'vcoco_eval/scripts/evaluate_vcoco_official.py'
    spec = importlib.util.spec_from_file_location('incom_v2_official_wrapper', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = module._load_evaluator(DATA / 'vsrl_eval.py')
    checked = {'images': 0, 'empty_images': 0, 'empty_case_checked': False}
    cls = audited_evaluator(base, checked)
    cls._audit_records = grouped
    module._load_evaluator = lambda path: cls
    pending = args.output_json.with_suffix('.pending.json')
    sys.argv = [str(source), '--cache', str(args.cache), '--vsrl-json', str(DATA / 'data/vcoco/vcoco_test.json'),
                '--coco-json', str(DATA / 'data/instances_vcoco_all_2014.json'),
                '--split-ids', str(DATA / 'data/splits/vcoco_test.ids'), '--evaluator', str(DATA / 'vsrl_eval.py'),
                '--output-json', str(pending), '--agent-ap']
    module.main()
    if checked['images'] != 4946:
        raise ValueError('Exhaustive collector verification did not run')
    metrics = json.loads(pending.read_text())
    metrics.update({'exhaustive_collector_equivalence': checked,
                    'collector_audit_source_sha256': digest(Path(__file__)),
                    'evidence_type': 'INDEPENDENT_PAPER_CORE_WITH_OUR_VCOCO_ADAPTATION',
                    'faithful_author_reproduction_verified': False})
    tmp = args.output_json.with_suffix('.partial')
    tmp.write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    tmp.replace(args.output_json)


if __name__ == '__main__':
    main()
