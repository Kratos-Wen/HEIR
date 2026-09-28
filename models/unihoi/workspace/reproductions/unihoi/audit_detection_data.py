"""Audit every training example before choosing a sequence/memory protocol."""

import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from .detection import BASE, ANNOTATIONS, DetectionExamples
from .tokenizer import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = DetectionExamples(AutoTokenizer.from_pretrained(BASE, local_files_only=True), max_length=1000000)
    lengths, counts, per_channel, over = [], Counter(), Counter(), {n: [] for n in (2048, 4096, 8192)}
    for i in range(len(data)):
        item = data[i]
        length = item['input_ids'].shape[1]
        lengths.append(length)
        events = json.loads(item['answer'])
        counts['events'] += len(events)
        counts['empty_event_images'] += not bool(events)
        for event in events:
            if not event['roles']:
                counts['agent_only_events'] += 1
            if set(event['roles']) == {'obj', 'instr'}:
                counts['dual_role_events'] += 1
            for role, entity in event['roles'].items():
                counts['missing_roles' if entity is None else 'visible_roles'] += 1
                per_channel[event['action'] + '_' + role] += 1
        for threshold in over:
            if length > threshold:
                over[threshold].append(item['image_id'])
        if (i+1) % 1000 == 0:
            print(f'validated training records {i+1}/{len(data)}', flush=True)
    ordered = sorted(lengths)
    result = {'schema': 'unihoi_train_serialization_audit_1', 'images': len(data),
        'annotation_sha256': digest(ANNOTATIONS), 'test_labels_read': False,
        'events_truncated': False, 'images_dropped': False, 'counts': dict(counts),
        'per_action_role': dict(per_channel), 'sequence_length_min': min(lengths),
        'sequence_length_max': max(lengths),
        'sequence_length_percentiles': {str(p): ordered[round((len(ordered)-1)*p/100)] for p in (50, 90, 95, 99)},
        'over_limit_image_ids': {str(k): v for k, v in over.items()},
        'sequence_lengths': lengths, 'image_ids': [r['image_id'] for r in data.rows]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.partial')
    tmp.write_text(json.dumps(result, indent=2) + '\n')
    tmp.replace(args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in
        ('sequence_lengths', 'image_ids', 'over_limit_image_ids')}), flush=True)


if __name__ == '__main__':
    main()
