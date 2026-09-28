"""Independent ranked-score and native role export, fixed before training."""

import math

import torch

from .diffusion import decode, initialize


@torch.no_grad()
def predict_records(model, cache, channels, image_id, width, height, device):
    if not len(cache['appearance']):
        return []
    results = []
    for start in range(0, len(cache['appearance']), 32):
        prior = initialize(cache['prior'][start:start+32].to(device), len(channels))
        appearance = cache['appearance'][start:start+32].to(device)
        results.append(model.reverse(prior, appearance, steps=50).cpu())
    images = torch.cat(results)
    nouns, _, joint = decode(images, cache['entity_query'])
    null = cache['null']
    # Virtual missing entities retain the null noun rather than borrowing a
    # real object class during cross-person pooling.
    joint[null] = images[null, 80]
    scores = joint[..., 0] * cache['detection_score'][:, None]
    return records_from_scores(scores, cache, channels, image_id, width, height)


def records_from_scores(scores, cache, channels, image_id, width, height):
    if scores.shape != (len(cache['human_query']), len(channels)) or not torch.isfinite(scores).all():
        raise ValueError('Invalid inference score layout')
    scale = torch.tensor([width, height, width, height])
    rows = []
    for human in torch.unique(cache['human_query'], sorted=True):
        pair_ids = torch.where(cache['human_query'] == human)[0]
        row = {'image_id': int(image_id), 'person_box': (cache['human_boxes'][pair_ids[0]] * scale).tolist()}
        for j, channel in enumerate(channels):
            action, role = channel.rsplit('_', 1)
            available = pair_ids[cache['null'][pair_ids]] if role == 'agent' else pair_ids
            if not len(available):
                raise ValueError('Every detected human needs its virtual-null pair')
            pair = available[scores[available, j].argmax()]
            score = float(scores[pair, j])
            row[action + '_agent'] = max(row.get(action + '_agent', 0.), score)
            if role != 'agent':
                box = [math.nan] * 4 if cache['null'][pair] else (cache['entity_boxes'][pair] * scale).tolist()
                row[channel] = box + [score]
        rows.append(row)
    return rows
