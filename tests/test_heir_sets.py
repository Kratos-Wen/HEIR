import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.heir.core import average_precision, event_metrics
from evaluation.heir_sets import ground_truth, prediction


def fixtures():
    row = {'image_id': 'image', 'width': 100, 'height': 100,
           'boxes': [{'id': 1, 'category': 'person', 'bbox': [0, 0, 20, 50]},
                     {'id': 2, 'category': 'object', 'bbox': [50, 10, 70, 30]},
                     {'id': 3, 'category': 'object', 'bbox': [70, 60, 90, 80]}],
           'relations': [{'subject': 1, 'object': 2, 'verb': 'hold', 'role': 'target'}]}
    record = {'image_id': 'image', 'entities': [dict(id=box['id'], noun=box['category'],
               box=box['bbox'], score=.9) for box in row['boxes']],
              'sets': [dict(subject_id=1, action='hold', members=[dict(entity_id=2, role='target')], score=.8)]}
    return row, record


def score(row, record):
    return event_metrics({'image': ground_truth(row)}, {'image': prediction(record, row)}, .5, .5)['mAP']


def test_exact_sets_require_every_member_and_role():
    row, record = fixtures()
    assert score(row, record) == 1.
    record['sets'][0]['members'].append(dict(entity_id=3, role='target'))
    assert score(row, record) == 0.
    record['sets'][0]['members'] = [dict(entity_id=2, role='instrument')]
    assert score(row, record) == 0.


def test_shared_correspondence_is_one_to_one():
    row, record = fixtures()
    row['relations'].append(dict(subject=1, object=3, verb='hold', role='target'))
    record['sets'][0]['members'].append(dict(entity_id=3, role='target'))
    assert score(row, record) == 1.
    record['entities'][2]['box'] = record['entities'][1]['box']
    assert score(row, record) == 0.


def test_agent_only_scope_and_budget():
    row, record = fixtures()
    extra = copy.deepcopy(record['sets'][0])
    extra.update(action='touch', score=.9)
    record['sets'].append(extra)
    assert len(prediction(record, row)['events']) == 2
    row['evaluation'] = 'agent_only'
    assert len(prediction(record, row)['events']) == 1
    assert score(row, record) == 1.
    record['sets'] = record['sets'] * 51
    with pytest.raises(ValueError, match='100-set'):
        prediction(record, row)


def test_tied_confidences_are_grouped():
    assert average_precision([(.5, True), (.5, False)], 1) == .5
    assert average_precision([(.5, False), (.5, True)], 1) == .5


def test_set_cli_needs_no_model_or_training_data(tmp_path):
    row, record = fixtures()
    vocab = {key: [{'id': value} for value in values] for key, values in dict(
        nouns=['person', 'object'], verbs=['hold'],
        roles=['target', 'instrument', 'support', 'source', 'destination', 'constraint']).items()}
    annotation, vocabulary, predictions, output = [tmp_path / name for name in ('gt.json', 'vocab.json', 'pred.jsonl', 'metrics.json')]
    annotation.write_text(json.dumps({'images': [row]}))
    vocabulary.write_text(json.dumps(vocab))
    predictions.write_text(json.dumps(record) + '\n')
    result = subprocess.run([sys.executable, '-m', 'evaluation.heir_sets', '--annotations', str(annotation),
                             '--vocabulary', str(vocabulary), '--predictions', str(predictions), '--output', str(output)],
                            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['set_map_percent'] == 100.
