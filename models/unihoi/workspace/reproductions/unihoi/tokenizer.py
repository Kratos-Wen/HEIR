"""Strictly loaded released VQGAN and box-preserving UniHOI inference geometry."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / 'reference_repos/UniHOI'
ASSET = ROOT / 'assets/unihoi_reproduction/vqgan.ckpt'
SHA256 = '4ede986bf6b171db3081ce171ad88e4ac970793cea14c180b3e5ac5105f4cb43'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_vqgan(device):
    if digest(ASSET) != SHA256:
        raise ValueError('VQGAN asset hash mismatch')
    source = AUTHOR / 'data_process/vqgan/vqgan.py'
    spec = importlib.util.spec_from_file_location('unihoi_released_vqgan', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((AUTHOR / 'evaluation/chameleon/vqgan.yaml').read_text())['model']['params']
    params = {k: v for k, v in config.items() if k not in ('ckpt_path', 'lossconfig')}
    model = module.VQModel(**params)
    weights = torch.load(ASSET, map_location='cpu', weights_only=False)['state_dict']
    # Unlike the released strict=False path, every inference weight must match.
    extra = set(weights) - set(model.state_dict())
    # The pinned public checkpoint has two image-sized orphan parameters.
    # Neither released tokenizer implementation references custom_layer;
    # the author's strict=False loader also discards these exact entries.
    orphan = {'custom_layer.weight', 'custom_layer.bias'}
    if any(not k.startswith('loss.') and k not in orphan for k in extra):
        raise ValueError(f'Unexpected VQGAN inference keys: {sorted(extra)}')
    if any(tuple(weights[k].shape) != (3, 512, 512) for k in extra & orphan):
        raise ValueError('Unknown orphan VQGAN parameter shape')
    model.load_state_dict({k: v for k, v in weights.items() if k not in extra}, strict=True)
    return model.eval().requires_grad_(False).to(device), sorted(extra)


def letterbox(image, resolution=512):
    image = image.convert('RGB')
    width, height = image.size
    side = max(width, height)
    left, top = (side-width)//2, (side-height)//2
    padded = Image.new('RGB', (side, side), (122, 116, 104))
    padded.paste(image, (left, top))
    pixels = padded.resize((resolution, resolution), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(pixels).copy()).permute(2, 0, 1).float()/127.5-1
    return tensor, {'original_wh': [width, height], 'pad_left_top': [left, top],
                    'padded_side': side, 'resolution': resolution, 'scale': resolution/side}


def transform_boxes(boxes, geometry, inverse=False):
    shift = boxes.new_tensor(geometry['pad_left_top'] * 2)
    scale = geometry['scale']
    return boxes/scale-shift if inverse else (boxes+shift)*scale


@torch.no_grad()
def encode(model, tensor):
    _, _, (_, _, ids) = model.encode(tensor)
    ids = ids.reshape(len(tensor), -1)
    if ids.shape[1] != 1024 or ids.min() < 0 or ids.max() >= 8192:
        raise ValueError('Expected 32x32 tokens in the released 8192-entry codebook')
    return ids
