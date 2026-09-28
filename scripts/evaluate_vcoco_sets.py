"""Score complete native-slot assignments from a trusted V-COCO cache."""
import argparse
from collections import defaultdict
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.vcoco_sets import evaluate
from check_vcoco_coverage import verify


class CacheTemplate(defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)


class CacheUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'utils' and name == 'CacheTemplate':
            return CacheTemplate
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'coverage', 'vsrl-json', 'coco-json', 'split-ids', 'evaluator', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify(args.cache, args.coverage, args.split_ids)
    spec = importlib.util.spec_from_file_location('official_vcoco', args.evaluator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if 'bool' not in np.__dict__:
        np.bool = np.bool_
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = module.VCOCOeval(str(args.vsrl_json), str(args.coco_json), str(args.split_ids))
        database = evaluator._get_vcocodb()
    expected = [int(value) for value in args.split_ids.read_text().split()]
    if len(expected) != len(set(expected)) or len(database) != len(expected) or {
            int(entry['id']) for entry in database} != set(expected):
        raise ValueError('Evaluator image IDs do not match the declared split')
    with args.cache.open('rb') as handle:
        records = CacheUnpickler(handle).load()
    if not isinstance(records, list):
        raise ValueError('Expected an official list-format cache')
    by_image = defaultdict(list)
    for record in records:
        by_image[int(record['image_id'])].append(record)
    if not set(by_image).issubset(expected):
        raise ValueError('Predictions contain images outside the evaluation split')
    result = evaluate(database, by_image, evaluator.actions, evaluator.roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
