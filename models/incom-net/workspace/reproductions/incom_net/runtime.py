"""Shared data paths, batching and random-state helpers of the InCoM-Net recipe."""

import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .backbones import WORKSPACE

DATA = WORKSPACE.parent / 'data/v-coco'


def collate(batch):
    return tuple(map(list, zip(*batch)))


class RGBTest(Dataset):
    def __init__(self):
        self.ids = [int(x) for x in (DATA / 'data/splits/vcoco_test.ids').read_text().split()]
        if len(self.ids) != 4946 or len(set(self.ids)) != 4946:
            raise ValueError('Invalid official test ID set')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = Image.open(DATA / 'images/val2014' / f'COCO_val2014_{image_id:012d}.jpg').convert('RGB')
        original_wh = image.size
        image = TF.normalize(TF.to_tensor(TF.resize(image, 800, max_size=1333)),
                              [.485, .456, .406], [.229, .224, .225])
        return image, {'image_id': image_id, 'original_wh': original_wh}


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(rng):
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(rng['cuda'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
