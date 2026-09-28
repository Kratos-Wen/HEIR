"""Paper MFT correction and explicit shared-head null-candidate adaptation.

The zero-feature virtual entity is OUR V-COCO adapter, not an author claim.
ICR/ProCA, pair transformer and dual-source decoder retain the isolated core.
"""

import torch
from torch import nn
import torch.distributed as dist
from torchvision.ops import box_iou

from reproductions.incom_net.model import InCoMHead, human_entity_pairs, focal_mft_loss
from .data import CHANNELS, NUM_ROLES


class PaperMFTHead(InCoMHead):
    def reason(self, q, context, cnn, vlm, pairs, mode):
        if mode not in ('full', 'detector_only', 'vlm_only'):
            raise ValueError('Unknown MFT mode')
        if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError('Invalid pair shape/type')
        if pairs.numel() and ((pairs < 0).any() or (pairs >= len(q)).any() or (pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError('Invalid pair identity')
        if not len(pairs):
            features = q.new_empty((0, self.cfg.hidden_dim))
            return {'logits': self.classifier(features), 'pair_features': features}
        h, o = pairs.unbind(-1)
        use_d, use_v = mode != 'vlm_only', mode != 'detector_only'
        dq, vq = torch.cat((q[h], q[o]), -1), torch.cat((context[h], context[o]), -1)
        # Paper Eq.12 + Sec.3.4: mask INPUTS, retain affine/LN bias responses.
        d = self.detector_pair(dq if use_d else torch.zeros_like(dq))
        v = self.context_pair(vq if use_v else torch.zeros_like(vq))
        z = self.pair_encoder((d + v).unsqueeze(0))
        dmem = self.cnn_proj(cnn).unsqueeze(0) if use_d else None
        vmem = self.vlm_proj(vlm).unsqueeze(0) if use_v else None
        for layer in self.decoder:
            z = layer(z, dmem, vmem, use_d, use_v)
        return {'logits': self.classifier(z[0]), 'pair_features': z[0]}


def candidate_pairs(labels):
    real = human_entity_pairs(labels)
    people = torch.where(labels == 0)[0]
    null = torch.stack((people, torch.full_like(people, len(labels))), -1)
    return torch.cat((real, null))


def valid_channels(record, pairs, compatibility):
    valid = torch.zeros(len(pairs), len(CHANNELS), dtype=torch.bool, device=pairs.device)
    null = pairs[:, 1] == len(record['labels'])
    valid[~null] = compatibility[record['labels'][pairs[~null, 1]]]
    valid[null] = True
    return valid


def associate(record, pairs, target, compatibility):
    valid = valid_channels(record, pairs, compatibility)
    y = record['boxes'].new_zeros(valid.shape)
    if not len(pairs) or not len(target['boxes']):
        return y, valid
    # Like official evaluation, match each predicted person to its best GT person.
    quality, index = box_iou(record['boxes'][pairs[:, 0]], target['boxes']).max(-1)
    matched = quality >= .5
    known = target['labels'][index] >= 0
    valid[matched] &= known[matched]
    positive = (target['labels'][index] == 1) & matched[:, None]
    null = pairs[:, 1] == len(record['labels'])
    visible = target['visible'][index]
    y[null] = (positive & ~visible)[null].to(y.dtype)
    if (~null).any():
        predicted = record['boxes'][pairs[~null, 1]][:, None, :]
        gold = target['role_boxes'][index[~null]]
        intersection = (torch.minimum(predicted[..., 2:], gold[..., 2:]) -
                        torch.maximum(predicted[..., :2], gold[..., :2])).clamp(min=0).prod(-1)
        union = (predicted[..., 2:] - predicted[..., :2]).prod(-1) + (gold[..., 2:] - gold[..., :2]).clamp(min=0).prod(-1) - intersection
        overlap = intersection / union.clamp(min=1e-12)
        noun_match = record['labels'][pairs[~null, 1]][:, None] == target['objects'][index[~null]]
        y[~null] = (positive[~null] & visible[~null] & (overlap >= .5) & noun_match).to(y.dtype)
    y *= valid
    return y, valid


class CompleteDetector(nn.Module):
    def __init__(self, extractor, cfg, compatibility):
        super().__init__()
        if cfg.num_actions != len(CHANNELS) or compatibility.shape != (80, len(CHANNELS)):
            raise ValueError('Full original V-COCO channel space is required')
        self.extractor = extractor
        self.head = PaperMFTHead(cfg)
        self.register_buffer('compatibility', compatibility.bool())

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError('Training requires original annotations')
        records = self.extractor(images)
        modes = ('full', 'detector_only', 'vlm_only') if self.training else ('full',)
        logits, ys, masks, predictions = {m: [] for m in modes}, [], [], []
        for i, record in enumerate(records):
            pairs = candidate_pairs(record['labels'])
            context = self.head.mine_context(record['detector_layers'], record['vlm_layers'], record['normalized_boxes'], record['grid'])
            q = torch.cat((record['detector_layers'][-1], context.new_zeros((1, self.head.cfg.detector_dim))))
            context = torch.cat((context, context.new_zeros((1, self.head.cfg.hidden_dim))))
            branches = {m: self.head.reason(q, context, record['cnn_tokens'], record['vlm_layers'][-1], pairs, m) for m in modes}
            if self.training:
                y, mask = associate(record, pairs, targets[i], self.compatibility)
                ys.append(y)
                masks.append(mask)
                for m in modes:
                    logits[m].append(branches[m]['logits'])
            else:
                confidence = torch.cat((record['scores'], record['scores'].new_ones(1)))
                scores = branches['full']['logits'].sigmoid() * confidence[pairs].prod(-1, keepdim=True).pow(2.8)
                scores *= valid_channels(record, pairs, self.compatibility)
                predictions.append({'boxes': record['boxes'], 'pairs': pairs, 'scores': scores, 'size': record['size']})
        if not self.training:
            return predictions
        y, mask = torch.cat(ys), torch.cat(masks)
        normalizer = y.sum().detach()
        if dist.is_initialized():
            dist.all_reduce(normalizer)
            normalizer = normalizer.clamp(min=1) / dist.get_world_size()
        else:
            normalizer = normalizer.clamp(min=1)
        loss, terms = focal_mft_loss({m: {'logits': torch.cat(values)} for m, values in logits.items()}, y, mask,
                                    alpha=.5, gamma=.1, normalizer=normalizer)
        return {'loss': loss, 'matched_positive_edges': y.sum().detach(),
                **{f'mft_{m}': x.detach() for m, x in terms.items()}}


def export_vcoco(prediction, image_id, original_wh):
    boxes, pairs, scores = (prediction[k].detach().cpu() for k in ('boxes', 'pairs', 'scores'))
    h, w = prediction['size'].tolist()
    ow, oh = original_wh
    boxes = boxes * boxes.new_tensor([ow / w, oh / h, ow / w, oh / h])
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite export')
    output = []
    for human in pairs[:, 0].unique(sorted=True).tolist():
        selected = torch.where(pairs[:, 0] == human)[0]
        record = {'image_id': int(image_id), 'person_box': boxes[human].tolist()}
        for c, name in enumerate(CHANNELS):
            best = selected[scores[selected, c].argmax()]
            score = float(scores[best, c])
            if c < NUM_ROLES:
                obj = int(pairs[best, 1])
                # Official V-COCO accepts an all-zero box as a missing role.
                box = [0.] * 4 if obj == len(boxes) else boxes[obj].tolist()
                record[name] = box + [score]
                agent = name.rsplit('_', 1)[0] + '_agent'
                record[agent] = max(record.get(agent, 0.), score)
            else:
                record[name] = score
        output.append(record)
    return output
