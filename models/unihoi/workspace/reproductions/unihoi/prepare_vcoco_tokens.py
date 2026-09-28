"""Prepare resumable train-only VQ codes; this is not UniHOI model training.

Uses the released inference letterbox rather than dropping images or clipping
HOI boxes via the repository's generic image-caption center-crop script.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .tokenizer import ROOT, AUTHOR, SHA256, digest, letterbox, encode, load_vqgan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualify-only', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run in an existing idle GPU allocation')
    data = ROOT.parent / 'data/v-coco'
    ids_file = data / 'data/splits/vcoco_trainval.ids'
    ids = [int(x) for x in ids_file.read_text().split()]
    if len(ids) != 5400 or len(set(ids)) != 5400:
        raise ValueError('Invalid trainval IDs')
    model, ignored = load_vqgan(torch.device('cuda', 0))
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'stage': 'TRAIN_IMAGE_TOKEN_PREPARATION_NOT_MODEL_TRAINING',
                'asset_sha256': SHA256, 'train_ids_sha256': digest(ids_file),
                'source_sha256': {str(f): digest(f) for f in (
                    Path(__file__), Path(__file__).with_name('tokenizer.py'),
                    AUTHOR / 'data_process/vqgan/vqgan.py', AUTHOR / 'evaluation/chameleon/vqgan.yaml')},
                'geometry': 'author inference pad-to-square 122/116/104 then LANCZOS 512',
                'geometry_not_author_training_verified': True,
                'ignored_non_model_keys': ignored, 'test_labels_read': False,
                'llm_vocabulary_offset_applied': False}
    sealed = args.output / 'protocol.json'
    if sealed.exists() and json.loads(sealed.read_text()) != protocol:
        raise ValueError('Different preprocessing protocol; use a new output')
    if not sealed.exists():
        sealed.write_text(json.dumps(protocol, indent=2)+'\n')
    selected = ids[:2] if args.qualify_only else ids
    manifest = []
    for index, image_id in enumerate(selected):
        path = data / 'images/train2014' / f'COCO_train2014_{image_id:012d}.jpg'
        image_hash = digest(path)
        output = args.output / f'{image_id:012d}.npz'
        meta = args.output / f'{image_id:012d}.json'
        if output.exists() and meta.exists():
            record = json.loads(meta.read_text())
            with np.load(output, allow_pickle=False) as arrays:
                tokens = arrays['vq_codes']
                if tokens.shape != (1024,) or tokens.min() < 0 or tokens.max() >= 8192:
                    raise ValueError('Corrupt cached codes')
            if record['image_sha256'] != image_hash or record['codes_sha256'] != digest(output):
                raise ValueError('Cached image/code mismatch')
        else:
            with Image.open(path) as image:
                tensor, geometry = letterbox(image)
            tokens = encode(model, tensor[None].cuda())[0].cpu().numpy().astype(np.uint16)
            with output.with_suffix('.partial').open('wb') as f:
                np.savez_compressed(f, vq_codes=tokens)
            output.with_suffix('.partial').replace(output)
            record = {'image_id': image_id, 'image_sha256': image_hash, 'geometry': geometry,
                      'codes_path': output.name, 'codes_sha256': digest(output)}
            meta.with_suffix('.partial').write_text(json.dumps(record, indent=2)+'\n')
            meta.with_suffix('.partial').replace(meta)
        manifest.append(record)
        if index % 100 == 0 or index == len(selected)-1:
            print(f'train-only VQ tokens {index+1}/{len(selected)} peak_mib={torch.cuda.max_memory_allocated()/1024**2:.1f}', flush=True)
    result = {'stage': protocol['stage'], 'images': len(manifest), 'image_ids': selected,
              'complete_trainval': len(manifest) == 5400,
              'manifest': manifest, 'test_read': False}
    name = 'qualification.json' if args.qualify_only else 'complete_manifest.json'
    (args.output / name).with_suffix('.partial').write_text(json.dumps(result, indent=2)+'\n')
    (args.output / name).with_suffix('.partial').replace(args.output / name)


if __name__ == '__main__':
    main()
