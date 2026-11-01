"""Original-label V-COCO ingestion; training consumes a train-only artifact."""

from collections import Counter
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, functional as TF

from reproductions.incom_net.vcoco import ACTION_NAMES, AGENT_ONLY, SCALES, square_crop_region

ROLE_NAMES = ACTION_NAMES + ('point_instr',)
CHANNELS = ROLE_NAMES + tuple(name + '_agent' for name in AGENT_ONLY)
NUM_ROLES = len(ROLE_NAMES)


def build_annotations(vsrl, coco, split_ids):
    """No test VSRL input: retain unknown people and annotated missing objects."""
    ids = list(map(int, split_ids))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicated split IDs')
    selected = set(ids)
    images = {r['id']: r for r in coco['images'] if r['id'] in selected}
    if set(images) != selected:
        raise ValueError('Missing COCO image metadata')
    categories = {c['id']: i for i, c in enumerate(sorted(coco['categories'], key=lambda x: x['id']))}
    annotations = {a['id']: a for a in coco['annotations'] if a['image_id'] in selected}
    people, per_image = {}, {i: [] for i in ids}

    def box(a):
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0:
            raise ValueError('Invalid annotated box')
        return [x, y, x + w, y + h]

    for a in annotations.values():
        if a['category_id'] == 1:
            person = {'annotation_id': a['id'], 'box': box(a), 'labels': [-1] * len(CHANNELS),
                      'objects': [-1] * len(CHANNELS), 'role_boxes': [[0.] * 4 for _ in CHANNELS],
                      'visible': [False] * len(CHANNELS)}
            people[a['id']] = person
            per_image[a['image_id']].append(person)
    seen_actions = set()
    for action in vsrl:
        name = action['action_name']
        if name in seen_actions:
            raise ValueError('Duplicate action definition')
        seen_actions.add(name)
        n = len(action['ann_id'])
        if (len(action['role_object_id']) != n * len(action['role_name'])
                or len(action['label']) != n or len(action['image_id']) != n):
            raise ValueError('Misaligned original role matrix')
        roles = action['role_name']
        if not roles or roles[0] != 'agent':
            raise ValueError('Agent must be the first original role')
        for i, (image_id, ann_id, label) in enumerate(zip(action['image_id'], action['ann_id'], action['label'])):
            if image_id not in selected or ann_id not in people:
                raise ValueError('VSRL source contains an out-of-split or non-person subject')
            if annotations[ann_id]['image_id'] != image_id or action['role_object_id'][i] != ann_id:
                raise ValueError('VSRL/COCO subject identity mismatch')
            if label not in (-1, 0, 1):
                raise ValueError('Unknown action label')
            person = people[ann_id]
            slots = [(0, 'agent')] if len(roles) == 1 else list(enumerate(roles))[1:]
            for r, role in slots:
                channel = CHANNELS.index(name + '_' + role)
                person['labels'][channel] = label
                object_id = action['role_object_id'][r * n + i] if role != 'agent' else 0
                if label == 1 and object_id:
                    obj = annotations[object_id]
                    if obj['image_id'] != image_id:
                        raise ValueError('Role object from another image')
                    person['visible'][channel] = True
                    person['role_boxes'][channel] = box(obj)
                    person['objects'][channel] = categories[obj['category_id']]
    expected_actions = {name.rsplit('_', 1)[0] for name in CHANNELS}
    if seen_actions != expected_actions:
        raise ValueError('Incomplete official action vocabulary')
    compatibility = [[False] * len(CHANNELS) for _ in categories]
    counts = Counter()
    for p in people.values():
        for c, label in enumerate(p['labels']):
            if label != 1:
                continue
            kind = 'agent' if c >= NUM_ROLES else 'visible' if p['visible'][c] else 'null'
            counts[kind] += 1
            counts[kind + ':' + CHANNELS[c]] += 1
            if kind == 'visible':
                compatibility[p['objects'][c]][c] = True
    rows = [{'image_id': i, 'file_name': images[i]['file_name'], 'people': per_image[i]} for i in ids]
    return {'schema': 'incom_complete_vcoco_train_1', 'channels': list(CHANNELS),
            'image_ids': ids, 'annotations': rows, 'compatibility': compatibility,
            'counts': dict(sorted(counts.items())), 'compatibility_source': 'positive visible train edges only'}


def crop_targets(target, top, left, side):
    target = {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}
    offset = target['boxes'].new_tensor([left, top, left, top])
    target['boxes'] = (target['boxes'] - offset).clamp(0, side)
    target['role_boxes'] = (target['role_boxes'] - offset).clamp(0, side)
    retained_roles = (target['role_boxes'][..., 2:] > target['role_boxes'][..., :2]).all(-1)
    # Cropping a visible object out is NOT an original missing-object annotation.
    target['labels'][target['visible'] & ~retained_roles] = -1
    keep = (target['boxes'][:, 2:] > target['boxes'][:, :2]).all(-1)
    for k in ('boxes', 'role_boxes', 'labels', 'visible', 'objects', 'person_ids'):
        target[k] = target[k][keep]
    return target


class CompleteVCOCO(Dataset):
    def __init__(self, path, image_dir, augment=True):
        data = json.loads(Path(path).read_text())
        if data['schema'] != 'incom_complete_vcoco_train_1' or tuple(data['channels']) != CHANNELS:
            raise ValueError('Incorrect training artifact schema/class order')
        self.annotations = data['annotations']
        self.image_dir, self.augment = Path(image_dir), augment
        self.compatibility = torch.tensor(data['compatibility'], dtype=torch.bool)
        self.jitter = ColorJitter(.4, .4, .4)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations[index]
        image = Image.open(self.image_dir / row['file_name']).convert('RGB')
        people = row['people']
        target = {'image_id': row['image_id'],
                  'boxes': torch.tensor([p['box'] for p in people], dtype=torch.float32).reshape(-1, 4),
                  'role_boxes': torch.tensor([p['role_boxes'] for p in people], dtype=torch.float32).reshape(-1, len(CHANNELS), 4),
                  'labels': torch.tensor([p['labels'] for p in people], dtype=torch.long).reshape(-1, len(CHANNELS)),
                  'visible': torch.tensor([p['visible'] for p in people], dtype=torch.bool).reshape(-1, len(CHANNELS)),
                  'objects': torch.tensor([p['objects'] for p in people], dtype=torch.long).reshape(-1, len(CHANNELS)),
                  'person_ids': torch.tensor([p['annotation_id'] for p in people], dtype=torch.long)}

        def resize(size):
            nonlocal image
            w, h = image.size
            image = TF.resize(image, size, max_size=1333)
            scale = torch.tensor([image.width / w, image.height / h] * 2)
            target['boxes'] *= scale
            target['role_boxes'] *= scale

        if self.augment:
            if random.random() < .5:
                image = TF.hflip(image)
                for key in ('boxes', 'role_boxes'):
                    b = target[key]
                    target[key] = b[..., [2, 1, 0, 3]] * torch.tensor([-1., 1., -1., 1.]) + torch.tensor([image.width, 0., image.width, 0.])
            image = self.jitter(image)
            resize(random.choice(SCALES))
            if random.random() < .5:
                top, left, side = square_crop_region(image.width, image.height)
                image = TF.crop(image, top, left, side, side)
                target = crop_targets(target, top, left, side)
                resize(random.choice(SCALES))
        else:
            resize(800)
        # Keep sentinel boxes invariant under image geometry operations.
        target['role_boxes'][~target['visible']] = 0
        target['size'] = torch.tensor([image.height, image.width])
        return TF.normalize(TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225]), target
