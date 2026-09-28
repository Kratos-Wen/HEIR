import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scorer'))

from evaluate_heir_sets import map_event, load_records


class ScoringInterfaces(unittest.TestCase):
    def test_set_loader_retains_dense_relation_output(self):
        record = {'image_id': 'test', 'boxes': [[0, 0, 10, 10], [20, 20, 30, 30]],
                  'labels': [0, 1], 'box_scores': [1.0, 1.0],
                  'hois': [{'subject_id': 0, 'object_id': 1, 'category_id': 0,
                            'score': 1.0 - i / 2000} for i in range(1001)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            path.write_text(json.dumps(record) + '\n')
            self.assertEqual(len(load_records(path, [['hold', 'target']], None)[0]['relations']), 1001)
            self.assertEqual(len(load_records(path, [['hold', 'target']], 100)[0]['relations']), 100)

    def test_map_decodes_the_independent_edge_product(self):
        single = map_event([('rel_0', 'ident_1', 'target', 0.3)])
        self.assertEqual([ids for ids, _ in single], [['rel_0']])
        self.assertAlmostEqual(single[0][1], 0.3)
        pair = map_event([('rel_0', 'ident_1', 'target', 0.8), ('rel_1', 'ident_2', 'target', 0.6),
                            ('rel_2', 'ident_2', 'instrument', 0.3)])
        scores = dict((tuple(ids), score) for ids, score in pair)
        self.assertAlmostEqual(scores['rel_0', 'rel_1'], 0.8 * 0.6)
        self.assertAlmostEqual(scores['rel_0', 'rel_2'], 0.8 * 0.3)
        self.assertAlmostEqual(scores[('rel_1',)], (1 - 0.8) * 0.6)      # one winner per (count, role-count) state
        self.assertAlmostEqual(scores[('rel_2',)], (1 - 0.8) * 0.3)
        self.assertEqual(len(pair), 4)
        self.assertEqual([score for _, score in pair], sorted((score for _, score in pair), reverse=True))

    def test_role_hoi_and_set_cli_with_synthetic_dataset(self):
        roles = ['target', 'instrument', 'support', 'source', 'destination', 'constraint']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'annotations').mkdir()
            vocab = {'version': 'test', 'verbs': [{'id': 'hold'}],
                     'nouns': [{'id': noun, 'l1': noun} for noun in ('person', 'cup')],
                     'roles': [{'id': role} for role in roles]}
            (root / 'vocabulary.json').write_text(json.dumps(vocab))
            row = {'image_id': 'sample', 'file_name': 'sample.jpg', 'width': 100, 'height': 100,
                   'boxes': [{'id': 1, 'category': 'person', 'bbox': [0, 0, 20, 20]},
                             {'id': 2, 'category': 'cup', 'bbox': [30, 30, 40, 40]}],
                   'relations': [{'subject': 1, 'object': 2, 'verb': 'hold', 'role': 'target'}]}
            for split in ('train', 'test'):
                (root / 'annotations' / f'{split}.json').write_text(json.dumps({'split': split, 'images': [row]}))
            (root / 'classes.json').write_text(json.dumps({'objects': ['person', 'cup'],
                                                          'interaction_verb_role': [['hold', 'target']]}))
            pred = {'image_id': 'sample', 'boxes': [[0, 0, 20, 20], [30, 30, 40, 40]],
                    'labels': [0, 1], 'box_scores': [1, 1],
                    'hois': [{'subject_id': 0, 'object_id': 1, 'category_id': 0, 'score': .9}]}
            (root / 'predictions.jsonl').write_text(json.dumps(pred) + '\n')
            for name, filename, extra in [('role', 'evaluate_heir_predictions.py', []),
                                           ('hoi', 'evaluate_heir_predictions.py', ['--hoi']),
                                           ('set', 'evaluate_heir_sets.py', ['--construction', 'top-k'])]:
                output = root / name
                subprocess.run([sys.executable, str(ROOT / 'scorer' / filename),
                                '--heir-root', str(root), '--classes', str(root / 'classes.json'),
                                '--split', 'test', '--predictions', str(root / 'predictions.jsonl'),
                                '--output-dir', str(output), *extra], check=True, capture_output=True, text=True)
                scores = json.loads((output / 'summary.json').read_text())
                self.assertAlmostEqual(scores['set_mAP' if name == 'set' else 'mAP'], 1.0)


if __name__ == '__main__':
    unittest.main()
