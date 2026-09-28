import json
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with path.open('rb') as handle:
        return tomllib.load(handle)


class Configurations(unittest.TestCase):
    def test_every_configuration_resolves(self):
        registry = load(ROOT / 'configs/checkpoints.toml')
        configs = sorted((ROOT / 'configs').glob('*/*.toml'))
        self.assertTrue(configs)
        for path in configs:
            config = load(path)
            self.assertIn(config['checkpoint'], registry, path)
            if config.get('family') == 'corisp':
                continue
            if config.get('benchmark') == 'vcoco':
                self.assertEqual(len(config['commit']), 40, path)
                self.assertTrue(config['checkpoint_asset'], path)
                continue
            commands = json.loads((ROOT / 'models' / config['model'] / 'baseline.json').read_text())['commands']
            for stage in ('prepare', 'support', 'export', 'score-role', 'score-hoi', 'score-sets-topk', 'score-sets-map'):
                self.assertIn(f"{config['variant']}-{stage}", commands, path)
            needed = set(commands[f"{config['variant']}-export"]['assets']) - {'HEIR', 'HEIR_SUPPORT'}
            self.assertEqual(needed, set(config.get('assets', [])), path)

    def test_registry_entries_are_pinned_or_marked_todo(self):
        for key, entry in load(ROOT / 'configs/checkpoints.toml').items():
            if entry.get('released') is False or entry['sha256'].startswith('TODO'):
                continue
            self.assertEqual(len(entry['sha256']), 64, key)
            self.assertTrue(entry['file'].endswith(('.pth', '.pt', '.json', '.csv')), key)


if __name__ == '__main__':
    unittest.main()
