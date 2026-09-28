"""Preserve physical IDs and observed event completeness through augmentation."""
from collections import defaultdict
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.ops import box_convert

from .environment import ROOT
from datasets.hico import make_hico_transforms
from heir_training.data import noun_order


def observed_target(row, transformed, verbs, roles, nouns):
    if row.get('evaluation') not in (None, 'agent_only'):
        raise ValueError('Unknown observation scope')
    ids = transformed['labels'][:, 0].long()
    kept = {int(identity): i for i, identity in enumerate(ids)}
    original = {b['id']: b for b in row['boxes']}
    if len(original) != len(row['boxes']) or len(kept) != len(ids):
        raise ValueError('Non-unique physical entity IDs')
    people = [i for i in kept if original[i]['category'] == 'person']
    person_index = {identity: i for i, identity in enumerate(people)}
    partial = row.get('evaluation') == 'agent_only'
    labels = torch.full((len(people), len(verbs)), -1 if partial else 0, dtype=torch.long)
    groups = defaultdict(list)
    occurrences = set()
    for edge in row['relations']:
        key = (edge['subject'], edge['object'], edge['verb'])
        if key in occurrences or edge['subject'] == edge['object']:
            raise ValueError('Duplicate/conflicting relation or self loop')
        if original[edge['subject']]['category'] != 'person':
            raise ValueError('Only a full person can be a subject')
        occurrences.add(key)
        groups[(edge['subject'], verbs[edge['verb']])].append(edge)
    visible = []
    crop_hidden_events = 0
    for (subject, action), edges in groups.items():
        if subject not in person_index:
            continue
        if any(e['object'] not in kept for e in edges):
            # Never relabel a cropped positive set as a smaller complete set.
            labels[person_index[subject], action] = -1
            crop_hidden_events += 1
            continue
        labels[person_index[subject], action] = 1
        visible.extend((subject, e['object'], action, roles[e['role']]) for e in edges)
    size = transformed['size']
    scale = torch.tensor([size[1], size[0], size[1], size[0]], dtype=torch.float32)
    boxes = box_convert(transformed['boxes'], 'cxcywh', 'xyxy') * scale
    return {'image_id': row['image_id'], 'size': size, 'orig_size': transformed['orig_size'],
            'instance_ids': ids, 'boxes': boxes,
            'categories': torch.tensor([nouns[original[int(i)]['category']] for i in ids], dtype=torch.long),
            'person_ids': torch.tensor(people, dtype=torch.long),
            'person_boxes': boxes[[kept[p] for p in people]],
            'action_observed': labels,
            'relations': torch.tensor(visible, dtype=torch.long).reshape(-1, 4),
            'partial': partial, 'crop_hidden_events': crop_hidden_events}


class HEIREvents(Dataset):
    def __init__(self, root, training=True, split='train'):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Expected train, val or test split')
        if training and split != 'train':
            raise ValueError('Training augmentation requires the training split')
        self.root = Path(root)
        self.vocabulary = json.loads((self.root/'vocabulary.json').read_text())
        self.nouns = noun_order(self.vocabulary)
        self.verbs = [x['id'] for x in self.vocabulary['verbs']]
        self.roles = [x['id'] for x in self.vocabulary['roles']]
        self.rows = json.loads((self.root/f'annotations/{split}.json').read_text())['images']
        self.transform = make_hico_transforms('train' if training else 'val')
        self.lookup = [{s: i for i, s in enumerate(values)} for values in (self.verbs, self.roles, self.nouns)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root/row['file_name']) as handle:
            image = handle.convert('RGB')
        if image.size != (row['width'], row['height']):
            raise ValueError('Image geometry changed')
        boxes = torch.tensor([b['bbox'] for b in row['boxes']], dtype=torch.float32).reshape(-1, 4)
        target = {'boxes': boxes, 'labels': torch.tensor([[b['id'], self.lookup[2][b['category']]] for b in row['boxes']], dtype=torch.long),
                  'area': (boxes[:, 2:]-boxes[:, :2]).prod(-1), 'iscrowd': torch.zeros(len(boxes), dtype=torch.long),
                  'orig_size': torch.tensor([row['height'], row['width']]), 'size': torch.tensor([row['height'], row['width']])}
        image, transformed = self.transform(image, target)
        return image, observed_target(row, transformed, *self.lookup)


def collate(batch):
    return tuple(map(list, zip(*batch)))
