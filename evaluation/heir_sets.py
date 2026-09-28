"""Evaluate submitted native HEIR sets with shared image-level identities."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .heir.core import box_iou, event_metrics
from .heir.schema import validate_image


def ground_truth(row):
    identifiers = {box['id']: f'box_{j + 1}' for j, box in enumerate(row['boxes'])}
    if len(identifiers) != len(row['boxes']):
        raise ValueError('Duplicate ground-truth identity')
    entities = [{'id': identifiers[box['id']], 'noun': box['category'], 'bbox': box['bbox']}
                for box in row['boxes']]
    relations = [{'id': f'rel_{j}', 'subject_id': identifiers[edge['subject']],
                  'entity_id': identifiers[edge['object']], 'verb': edge['verb'], 'role': edge['role']}
                 for j, edge in enumerate(row['relations'])]
    groups = defaultdict(list)
    for edge in relations:
        groups[edge['subject_id'], edge['verb']].append(edge['id'])
    events = [{'id': f'set_{j}', 'verb': key[1], 'relation_ids': sorted(members)}
              for j, (key, members) in enumerate(sorted(groups.items()))]
    return {'id': row['image_id'], 'width': row['width'], 'height': row['height'],
            'entities': entities, 'relations': relations, 'events': events, 'events_complete': True,
            'coverage': 'adjudicated_closed_world', 'cluster_id': row['image_id']}


def prediction(record, row):
    if record['image_id'] != row['image_id']:
        raise ValueError('Prediction and annotation image IDs differ')
    if len(record['sets']) > 100:
        raise ValueError('Native predictions must already satisfy the 100-set image budget')
    entities = []
    for entity in record['entities']:
        x1, y1, x2, y2 = entity['box']
        x1, x2 = max(0., min(row['width'], x1)), max(0., min(row['width'], x2))
        y1, y2 = max(0., min(row['height'], y1)), max(0., min(row['height'], y2))
        if x1 < x2 and y1 < y2:
            entities.append({'id': f"node_{entity['id']}", 'noun': entity['noun'],
                             'bbox': [x1, y1, x2, y2], 'score': entity['score']})
    nodes = {entity['id']: entity for entity in entities}
    boxes = {box['id']: box['bbox'] for box in row['boxes']}
    scope = {(edge['subject'], edge['verb']) for edge in row['relations']}
    if row.get('evaluation') not in (None, 'agent_only'):
        raise ValueError('Unknown annotation scope')
    edges, hypotheses = {}, []
    for item in record['sets']:
        subject, verb = f"node_{item['subject_id']}", item['action']
        if subject not in nodes:
            continue
        if row.get('evaluation') == 'agent_only' and not any(
                action == verb and box_iou(nodes[subject]['bbox'], boxes[actor]) >= .5
                for actor, action in scope):
            continue
        members = [(subject, f"node_{member['entity_id']}", verb, member['role'])
                   for member in item['members']]
        if not members or any(entity not in nodes or entity == subject for _, entity, _, _ in members):
            continue
        if len(set(members)) != len(members):
            raise ValueError('Duplicate member within a predicted set')
        score = float(item['score'])
        if not 0 <= score <= 1:
            raise ValueError('Invalid set confidence')
        for key in members:
            edges[key] = max(edges.get(key, 0.), score)
        hypotheses.append((members, score))
    relations, identifiers = [], {}
    for key, score in sorted(edges.items()):
        identifiers[key] = f'rel_{len(relations)}'
        relations.append({'id': identifiers[key], 'subject_id': key[0], 'entity_id': key[1],
                          'verb': key[2], 'role': key[3], 'score': score})
    events = [{'verb': members[0][2], 'relation_ids': sorted(identifiers[key] for key in members),
               'score': score} for members, score in hypotheses]
    events.sort(key=lambda event: (-event['score'], event['verb'], event['relation_ids']))
    for j, event in enumerate(events):
        event['id'] = f'set_{j}'
    return {'id': record['image_id'], 'entities': entities, 'relations': relations, 'events': events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    vocabulary = json.loads(args.vocabulary.read_text())
    taxonomy = {key: [item['id'] for item in vocabulary[key]] for key in ('verbs', 'nouns', 'roles')}
    rows = json.loads(args.annotations.read_text())['images']
    by_id = {row['image_id']: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError('Empty or duplicate evaluation image IDs')
    reference, predicted = {}, {}
    for row in rows:
        image = ground_truth(row)
        validate_image(image, taxonomy)
        reference[image['id']] = image
    with args.predictions.open() as handle:
        for line in handle:
            record = json.loads(line)
            identity = record['image_id']
            if identity not in by_id or identity in predicted:
                raise ValueError('Unknown or duplicate prediction image ID')
            row = by_id[identity]
            image = prediction(record, row)
            validate_image(image, taxonomy, prediction=True, dimensions=(row['width'], row['height']))
            predicted[identity] = image
    if set(predicted) != set(reference):
        raise ValueError('Predictions must include every image, including empty outputs')
    measured = event_metrics(reference, predicted, .5, .5)
    result = {'metric': 'HEIR_Set_mAP', 'set_map_percent': 100 * measured['mAP'],
              'images': len(reference), 'actions': len(measured['verbs']),
              'gt_sets': sum(row['gt_events'] for row in measured['verbs']),
              'per_action': [{'action': row['verb'], 'gt_sets': row['gt_events'],
                              'set_ap_percent': 100 * row['ap']} for row in measured['verbs']],
              'iou': .5, 'max_sets_per_image': 100,
              'ap_interpolation': 'all_points_precision_envelope_score_ties_grouped'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
