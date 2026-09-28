import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).with_name('baseline.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Contracts(unittest.TestCase):
    def test_reject_escape(self):
        for name in ('../outside', '/tmp/outside'):
            with self.assertRaises(ValueError):
                m.inside(name)

    def test_world_size(self):
        recipe = {'commands': {'train': {'argv': ['{python}', '-m', 'module'],
            'cwd': '{root}/workspace', 'world_size': 8, 'distributed': True}}}
        with self.assertRaises(ValueError):
            m.command_for(recipe, 'train', 'python', Path('/tmp/run'), 1, 4, 0, None, 29500)
        cmd, _, _ = m.command_for(recipe, 'train', 'python', Path('/tmp/run'), 2, 4, 1, 'node0', 29500)
        self.assertIn('--node-rank=1', cmd)
        self.assertEqual(cmd[-2:], ['-m', 'module'])

    def test_assets_no_clobber(self):
        previous = m.ROOT
        with tempfile.TemporaryDirectory() as temp:
            try:
                m.ROOT = Path(temp)
                (m.ROOT / 'asset').write_text('data')
                recipe = {'assets': {'A': {'mounts': [['link', '']]}}}
                m.link_assets(recipe, [f'A={temp}/asset'])
                m.link_assets(recipe, [f'A={temp}/asset'])
                (m.ROOT / 'other').write_text('other')
                with self.assertRaises(FileExistsError):
                    m.link_assets(recipe, [f'A={temp}/other'])
            finally:
                m.ROOT = previous


if __name__ == '__main__':
    unittest.main()
