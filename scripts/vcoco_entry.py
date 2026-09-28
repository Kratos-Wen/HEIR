"""Portable entry point; record visited RGBs even when predictions are empty."""
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


class RecordedDataset:
    def __init__(self, base):
        self.base, self.dataset, self.visited = base, base.dataset, []

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(index)
        sample = self.base[index]
        self.visited.append(int(self.dataset.image_id(index)))
        return sample


def main():
    import run_hdetr_corisp_v3_vcoco as trainer
    original = trainer._VCOCOTrainingDLE.cache_vcoco

    def record_cache(engine, loader, cache_dir):
        if engine.config.world_size != 1:
            raise ValueError('Official cache writer requires one rank')
        recorded = RecordedDataset(loader.dataset)
        original(engine, SimpleNamespace(dataset=recorded), cache_dir)
        expected = [int(x) for x in Path(engine.config.vcoco_eval_split_ids).read_text().split()]
        if len(recorded.visited) != len(expected) or set(recorded.visited) != set(expected):
            raise ValueError('Cache inference did not visit every official image exactly once')
        cache = Path(cache_dir)/'cache.pkl'
        with cache.open('rb') as handle:
            rows = pickle.load(handle)
        predicted = {int(row['image_id']) for row in rows}
        if not predicted.issubset(set(expected)):
            raise ValueError('Predictions contain out-of-split images')
        record = {'split': engine.config.partitions[1], 'visited_ids': recorded.visited,
            'images': len(expected), 'images_without_predictions': sorted(set(expected)-predicted),
            'cache_sha256': digest(cache), 'checkpoint_sha256': digest(engine.config.resume),
            'split_ids_sha256': digest(engine.config.vcoco_eval_split_ids)}
        path = Path(cache_dir)/'coverage.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(path)

    trainer._VCOCOTrainingDLE.cache_vcoco = record_cache
    trainer.main()


if __name__ == '__main__':
    main()
