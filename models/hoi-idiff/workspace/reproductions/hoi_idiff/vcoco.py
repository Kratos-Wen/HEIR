"""Native role-channel adaptation; unknown labels never become negatives."""

import json
from pathlib import Path

import torch
from torchvision.ops import box_iou

from .detector import ROOT

ANNOTATIONS = ROOT / 'vcoco_eval/runs/vcoco_baselines/incom_complete_adapter_seed42_20260919_r2/train_annotations.json'
DATA = ROOT.parent / 'data/v-coco'


def annotations():
    data = json.loads(ANNOTATIONS.read_text())
    ids = [int(x) for x in (DATA / 'data/splits/vcoco_trainval.ids').read_text().split()]
    rows = data['annotations']
    if len(rows) != 5400 or len(set(ids)) != 5400 or {r['image_id'] for r in rows} != set(ids):
        raise ValueError('Require complete official trainval artifact')
    if len(data['channels']) != 29 or data['channels'][24] != 'point_instr':
        raise ValueError('Unexpected role-channel order')
    return rows, data['channels']


def pair_targets(cache, row, width, height, channels):
    n, c = len(cache['human_boxes']), len(channels)
    labels = torch.full((n, c), -1, dtype=torch.int8)
    nouns = cache['predicted_noun'].clone()
    scale = torch.tensor([width, height, width, height])
    people = row['people']
    if n == 0 or not people:
        return labels, nouns
    hb = cache['human_boxes'] * scale
    ob = cache['entity_boxes'] * scale
    overlap = box_iou(hb, torch.tensor([p['box'] for p in people], dtype=torch.float32))
    quality, matched = overlap.max(-1)
    for i in range(n):
        if quality[i] < .5:
            continue
        person = people[int(matched[i])]
        best_quality = -1.
        for j, channel in enumerate(channels):
            truth = person['labels'][j]
            if truth == -1:
                continue
            agent = channel.endswith('_agent')
            if cache['null'][i]:
                labels[i, j] = truth if agent else int(truth == 1 and not person['visible'][j])
            elif agent:
                continue
            elif truth == 0 or not person['visible'][j]:
                labels[i, j] = 0
            else:
                iou = float(box_iou(ob[i:i+1], torch.tensor([person['role_boxes'][j]], dtype=torch.float32))[0, 0])
                if iou >= .5:
                    labels[i, j] = 1
                    if iou > best_quality:
                        nouns[i], best_quality = person['objects'][j], iou
                elif iou < .3:
                    labels[i, j] = 0
    return labels, nouns


def select_pairs(labels, epoch, image_id, maximum=8):
    generator = torch.Generator().manual_seed(42 + 1000003 * epoch + image_id)
    pos = torch.where((labels == 1).any(-1))[0]
    neg = torch.where((labels >= 0).any(-1) & ~(labels == 1).any(-1))[0]
    pos = pos[torch.randperm(len(pos), generator=generator)]
    neg = neg[torch.randperm(len(neg), generator=generator)]
    first = pos[:maximum // 2]
    negative = neg[:maximum - len(first)]
    remaining = pos[len(first):len(first) + maximum - len(first) - len(negative)]
    return torch.cat((first, negative, remaining))


def positive_coverage(cache, row, width, height, labels):
    expected = {(p['annotation_id'], j) for p in row['people']
                for j, value in enumerate(p['labels']) if value == 1}
    covered = set()
    if len(labels) and row['people']:
        scale = torch.tensor([width, height, width, height])
        overlaps = box_iou(cache['human_boxes'] * scale,
                          torch.tensor([p['box'] for p in row['people']], dtype=torch.float32))
        quality, person_ids = overlaps.max(-1)
        for pair, channel in torch.nonzero(labels == 1).tolist():
            if quality[pair] < .5:
                raise ValueError('Positive assigned without localized human')
            covered.add((row['people'][int(person_ids[pair])]['annotation_id'], channel))
    if not covered <= expected:
        raise ValueError('Candidate supervision introduces an unannotated positive')
    return {'original_positive_edges': len(expected), 'covered_positive_edges': len(covered),
            'missed_positive_edges': [list(x) for x in sorted(expected - covered)]}


def regression_loss(prediction, target, known):
    if prediction.shape != target.shape or known.shape != (len(prediction), prediction.shape[2]):
        raise ValueError('MSE target/mask mismatch')
    squared = (prediction - target).square().mean((1, 3))
    return (squared * known).sum(), known.sum()
