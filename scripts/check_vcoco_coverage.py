"""Require a complete, content-bound inference record before scoring."""
import argparse
import json
from pathlib import Path

from vcoco_entry import digest


def verify(cache, coverage, split_ids):
    record = json.loads(coverage.read_text())
    expected = [int(x) for x in split_ids.read_text().split()]
    if (record['cache_sha256'] != digest(cache) or record['split_ids_sha256'] != digest(split_ids)
            or len(record['visited_ids']) != len(expected) or set(record['visited_ids']) != set(expected)):
        raise ValueError('Incomplete or changed V-COCO inference cache')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'coverage', 'split-ids'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    record = verify(args.cache, args.coverage, args.split_ids)
    print(json.dumps({'passed': True, 'split': record['split'], 'images': record['images'],
                     'empty_images': len(record['images_without_predictions'])}))


if __name__ == '__main__':
    main()
