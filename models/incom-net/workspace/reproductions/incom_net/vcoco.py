"""Explicit native-24 V-COCO reconstruction contract, not HEIR Function-8.

The published InCoM-Net null convention is not available. This backend uses
visible pairs, declares that limitation, and does not invent a new null head.
"""

import json
from pathlib import Path
import random

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.ops import box_iou
from torchvision.transforms import ColorJitter, functional as TF
from PIL import Image

from .model import InCoMHead, human_entity_pairs, focal_mft_loss, inference_scores

ACTION_NAMES = ("hold_obj", "sit_instr", "ride_instr", "look_obj", "hit_instr", "hit_obj",
                "eat_obj", "eat_instr", "jump_instr", "lay_instr", "talk_on_phone_instr",
                "carry_obj", "throw_obj", "catch_obj", "cut_instr", "cut_obj",
                "work_on_computer_instr", "ski_instr", "surf_instr", "skateboard_instr",
                "drink_instr", "kick_obj", "read_obj", "snowboard_instr")
ALL_ROLES = ACTION_NAMES + ("point_instr",)
AGENT_ONLY = ("stand", "walk", "run", "smile")
SCALES = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)


def square_crop_region(width, height):
    if min(width, height) < 1:
        raise ValueError('Image dimensions must be positive')
    upper = min(600, width, height)
    # Long-side capping can reduce panorama short sides below 384 pixels.
    side = random.randint(min(384, upper), upper)
    return random.randint(0, height - side), random.randint(0, width - side), side


def crop_pair_targets(target, top, left, height, width):
    result = {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}
    for name in ("boxes_h", "boxes_o"):
        boxes = result[name] - result[name].new_tensor([left, top, left, top])
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        result[name] = boxes
    keep = ((result['boxes_h'][:, 2:] > result['boxes_h'][:, :2]).all(-1)
            & (result['boxes_o'][:, 2:] > result['boxes_o'][:, :2]).all(-1))
    for name in ("boxes_h", "boxes_o", "actions", "objects"):
        result[name] = result[name][keep]
    return result


def resize_pairs(image, target, size):
    old_w, old_h = image.size
    image = TF.resize(image, size, max_size=1333)
    new_w, new_h = image.size
    scale = torch.tensor([new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h])
    target = dict(target)
    for name in ("boxes_h", "boxes_o"):
        target[name] = target[name] * scale
    return image, target


class NativeVCOCO(Dataset):
    def __init__(self, annotation_file, image_dir, official_trainval_ids, augment=True):
        source = json.loads(Path(annotation_file).read_text())
        self.action_names = tuple(x.replace(" ", "_") for x in source['classes'])
        if self.action_names != ACTION_NAMES:
            raise ValueError("Unexpected native-24 class ordering")
        official_ids = {int(x) for x in Path(official_trainval_ids).read_text().split()}
        rows = source['annotations']
        parsed_ids = [int(Path(row['file_name']).stem.rsplit('_', 1)[1]) for row in rows]
        if len(set(parsed_ids)) != 5400 or set(parsed_ids) != official_ids:
            raise ValueError("Source annotations are not the complete official trainval split")
        self.annotations = [row for row in rows if len(row['actions'])]
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.jitter = ColorJitter(.4, .4, .4)
        self.compatibility = torch.zeros(80, len(ACTION_NAMES), dtype=torch.bool)
        for action, objects in enumerate(source['action_to_object']):
            for obj in objects:
                if not 1 <= obj <= 80:
                    raise ValueError("Expected one-based native COCO object IDs")
                self.compatibility[obj - 1, action] = True

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations[index]
        image = Image.open(self.image_dir / row['file_name']).convert('RGB')
        target = {'boxes_h': torch.tensor(row['boxes_h'], dtype=torch.float32).reshape(-1, 4),
                  'boxes_o': torch.tensor(row['boxes_o'], dtype=torch.float32).reshape(-1, 4),
                  'actions': torch.tensor(row['actions'], dtype=torch.long),
                  'objects': torch.tensor(row['objects'], dtype=torch.long) - 1,
                  'image_id': int(Path(row['file_name']).stem.rsplit('_', 1)[1])}
        if self.augment:
            if random.random() < .5:
                image = TF.hflip(image)
                width = image.width
                for name in ('boxes_h', 'boxes_o'):
                    box = target[name]
                    target[name] = box[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([width, 0, width, 0])
            image = self.jitter(image)
            image, target = resize_pairs(image, target, random.choice(SCALES))
            if random.random() < .5:
                top, left, side = square_crop_region(image.width, image.height)
                image = TF.crop(image, top, left, side, side)
                target = crop_pair_targets(target, top, left, side, side)
                image, target = resize_pairs(image, target, random.choice(SCALES))
        else:
            image, target = resize_pairs(image, target, 800)
        target['size'] = torch.tensor([image.height, image.width])
        image = TF.normalize(TF.to_tensor(image), [.485, .456, .406], [.229, .224, .225])
        return image, target


def associate_pairs(record, pairs, target, compatibility):
    labels = record['boxes'].new_zeros((len(pairs), len(ACTION_NAMES)))
    objects = record['labels'][pairs[:, 1]]
    valid = compatibility[objects]
    if not len(pairs) or not len(target['actions']):
        return labels, valid
    h, o = pairs.unbind(-1)
    overlap = torch.minimum(box_iou(record['boxes'][h], target['boxes_h']),
                            box_iou(record['boxes'][o], target['boxes_o']))
    # Explicit class-aware matching: a wrong predicted noun is not a positive HOI.
    matched = (overlap >= .5) & (objects[:, None] == target['objects'][None])
    rows, cols = matched.nonzero(as_tuple=True)
    labels[rows, target['actions'][cols]] = 1
    if (labels.bool() & ~valid).any():
        raise ValueError("Ground-truth edge missing from the train object-action compatibility table")
    return labels, valid


class InCoMDetector(nn.Module):
    def __init__(self, extractor, cfg, compatibility):
        super().__init__()
        if cfg.num_actions != len(ACTION_NAMES) or compatibility.shape != (80, len(ACTION_NAMES)):
            raise ValueError("This backend explicitly supports native-24 V-COCO")
        self.extractor = extractor
        self.head = InCoMHead(cfg)
        self.register_buffer('compatibility', compatibility.bool())

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Training requires real pair annotations")
        records = self.extractor(images)
        collected = {name: [] for name in ('full', 'detector_only', 'vlm_only')}
        all_labels, all_valid, detections = [], [], []
        for index, record in enumerate(records):
            pairs = human_entity_pairs(record['labels'])
            branches = self.head(record['detector_layers'], record['vlm_layers'], record['normalized_boxes'],
                                 record['grid'], record['cnn_tokens'], pairs)
            if self.training:
                labels, valid = associate_pairs(record, pairs, targets[index], self.compatibility)
                all_labels.append(labels)
                all_valid.append(valid)
                for name in collected:
                    collected[name].append(branches[name]['logits'])
            else:
                objects = record['labels'][pairs[:, 1]]
                valid = self.compatibility[objects]
                scores = inference_scores(branches['full']['logits'], record['scores'][pairs], valid)
                p, action = valid.nonzero(as_tuple=True)
                detections.append({'boxes': record['boxes'], 'pairing': pairs[p], 'scores': scores[p, action],
                                   'labels': action, 'objects': objects[p], 'size': record['size'],
                                   'pair_features': branches['full']['pair_features'], 'pair_indices': pairs})
        if not self.training:
            return detections
        labels, valid = torch.cat(all_labels), torch.cat(all_valid)
        normalizer = labels.sum().detach()
        if dist.is_initialized():
            dist.all_reduce(normalizer)
            # DDP averages gradients: clamp the GLOBAL count before dividing.
            normalizer = normalizer.clamp(min=1) / dist.get_world_size()
        else:
            normalizer = normalizer.clamp(min=1)
        branches = {name: {'logits': torch.cat(logits)} for name, logits in collected.items()}
        loss, terms = focal_mft_loss(branches, labels, valid, alpha=.5, gamma=.1, normalizer=normalizer)
        # Only `loss` participates in backward; diagnostics are detached.
        return {'loss': loss, 'matched_positive_edges': labels.sum().detach(),
                **{f'mft_{name}': term.detach() for name, term in terms.items()}}


def export_vcoco(prediction, image_id, original_wh):
    boxes = prediction['boxes'].detach().cpu()
    h, w = prediction['size'].detach().cpu().tolist()
    ow, oh = original_wh
    boxes = boxes * torch.tensor([ow / w, oh / h, ow / w, oh / h])
    records = []
    for pair, score, action in zip(prediction['pairing'].detach().cpu(), prediction['scores'].detach().cpu(),
                                   prediction['labels'].detach().cpu()):
        record = {'image_id': int(image_id), 'person_box': boxes[pair[0]].tolist()}
        for name in ALL_ROLES:
            record[name] = [0., 0., .1, .1, 0.]
            record[name.rsplit('_', 1)[0] + '_agent'] = 0.
        for name in AGENT_ONLY:
            record[name + '_agent'] = 0.
        name = ACTION_NAMES[int(action)]
        record[name] = boxes[pair[1]].tolist() + [float(score)]
        record[name.rsplit('_', 1)[0] + '_agent'] = float(score)
        records.append(record)
    return records
