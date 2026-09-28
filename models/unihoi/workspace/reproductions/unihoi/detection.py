"""Executable supervised detection path, not the full UniHOI cycle method.

Input-prefix IAA placement and JSON role serialization are explicit independent
adaptations. Only known input tokens may condition the visual prefix.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import PrefixInteractionAdapter
from .tokenizer import ROOT, digest

BASE = ROOT / 'assets/unihoi_reproduction/Meta-Llama-3-8B'
TOKENS = ROOT / 'vcoco_eval/runs/vcoco_baselines/unihoi_token_preparation_20260919'
ANNOTATIONS = ROOT / 'vcoco_eval/runs/vcoco_baselines/incom_complete_adapter_seed42_20260919_r2/train_annotations.json'
PROMPT = ('Describe the annotated human actions and their role objects as JSON. '
          'Each event has human, action, and roles. Each visible role has box and '
          'noun_id (zero-based COCO category order). Missing role objects are null. '
          'Boxes use xyxy coordinates on the padded image, scaled to 0..1000.\nImage: ')


def serialize_events(row, channels, geometry):
    side = geometry['padded_side']
    left, top = geometry['pad_left_top']

    def box(values):
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError('Invalid box')
        x1, y1, x2, y2 = values
        if not x1 < x2 or not y1 < y2:
            raise ValueError('Degenerate box')
        return [round(float(x) / side * 1000, 2)
                for x in (x1 + left, y1 + top, x2 + left, y2 + top)]

    events = []
    for person in sorted(row['people'], key=lambda p: p['annotation_id']):
        grouped = {}
        for i, channel in enumerate(channels):
            label = person['labels'][i]
            if label not in (-1, 0, 1):
                raise ValueError('Unknown supervision state')
            if label != 1:
                continue
            action, role = channel.rsplit('_', 1)
            event = grouped.setdefault(action, {'human': box(person['box']),
                                                'action': action, 'roles': {}})
            if role != 'agent':
                if person['visible'][i]:
                    noun = person['objects'][i]
                    if not 0 <= noun < 80:
                        raise ValueError('Invalid native noun ID')
                    event['roles'][role] = {'box': box(person['role_boxes'][i]), 'noun_id': noun}
                else:
                    event['roles'][role] = None
        events.extend(grouped[a] for a in sorted(grouped))
    return json.dumps(events, separators=(',', ':'), allow_nan=False)


class DetectionExamples:
    def __init__(self, tokenizer, annotation_path=ANNOTATIONS, token_dir=TOKENS, max_length=4096):
        data = json.loads(Path(annotation_path).read_text())
        if data['schema'] != 'incom_complete_vcoco_train_1':
            raise ValueError('Original-label training artifact required')
        ids_path = ROOT.parent / 'data/v-coco/data/splits/vcoco_trainval.ids'
        ids = [int(x) for x in ids_path.read_text().split()]
        if len(ids) != 5400 or set(ids) != set(data['image_ids']):
            raise ValueError('Not the official trainval split')
        self.rows, self.channels = data['annotations'], data['channels']
        if len(self.rows) != 5400 or {r['image_id'] for r in self.rows} != set(ids):
            raise ValueError('Incomplete or duplicate image rows')
        self.tokenizer, self.token_dir, self.max_length = tokenizer, Path(token_dir), max_length
        self.visual_start = len(tokenizer)
        if self.visual_start != 128256:
            raise ValueError('Unexpected Llama base vocabulary')
        self.prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
        self.separator = tokenizer.encode('\nAnswer: ', add_special_tokens=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        stem = f"{row['image_id']:012d}"
        meta = json.loads((self.token_dir / (stem + '.json')).read_text())
        path = self.token_dir / (stem + '.npz')
        if meta['image_id'] != row['image_id'] or digest(path) != meta['codes_sha256']:
            raise ValueError('Corrupt or mismatched VQ cache')
        with np.load(path, allow_pickle=False) as f:
            codes = f['vq_codes'].astype(np.int64)
        if codes.shape != (1024,) or codes.min() < 0 or codes.max() >= 8192:
            raise ValueError('Unexpected VQ code range/grid')
        answer = serialize_events(row, self.channels, meta['geometry'])
        prefix = ([self.tokenizer.bos_token_id] + self.prompt +
                  (codes + self.visual_start).tolist() + self.separator)
        target = self.tokenizer.encode(answer, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        ids = torch.tensor(prefix + target, dtype=torch.long)
        if len(ids) > self.max_length:
            raise ValueError(f'Image {row["image_id"]} needs {len(ids)} tokens; never truncate roles')
        labels = ids.clone()
        labels[:len(prefix)] = -100
        return {'input_ids': ids[None], 'attention_mask': torch.ones_like(ids[None]),
                'labels': labels[None], 'prefix_lengths': torch.tensor([len(prefix)]),
                'image_id': row['image_id'], 'answer': answer}

    def qualification_indices(self):
        candidates = {'multi_person': [], 'dual_role': [], 'missing_role': [], 'point': []}
        for i, row in enumerate(self.rows):
            active = [p for p in row['people'] if 1 in p['labels']]
            if len(active) > 1:
                candidates['multi_person'].append(i)
            for person in active:
                positives = [self.channels[j] for j, v in enumerate(person['labels']) if v == 1]
                if any(a + '_obj' in positives and a + '_instr' in positives
                       and person['visible'][self.channels.index(a + '_obj')]
                       and person['visible'][self.channels.index(a + '_instr')]
                       for a in ('cut', 'eat', 'hit')):
                    candidates['dual_role'].append(i)
                if 'point_instr' in positives:
                    candidates['point'].append(i)
                if any(v == 1 and self.channels[j].rsplit('_', 1)[1] != 'agent' and not person['visible'][j]
                       for j, v in enumerate(person['labels'])):
                    candidates['missing_role'].append(i)
        result = {}
        for kind, indices in candidates.items():
            if not indices:
                raise ValueError('Missing real training coverage: ' + kind)
            result[kind] = min(indices, key=lambda i: sum(p['labels'].count(1) for p in self.rows[i]['people']))
        return result


class DetectionLanguageModel(nn.Module):
    def __init__(self, llm, visual_start, visual_codes=8192, loss_chunk=64):
        super().__init__()
        self.llm, self.visual_start = llm, visual_start
        self.visual_codes, self.loss_chunk = visual_codes, loss_chunk
        if llm.config.vocab_size != visual_start + visual_codes:
            raise ValueError('Expand both language-model vocabulary matrices first')
        width = llm.config.hidden_size
        self.modality = nn.Embedding(2, width)
        nn.init.xavier_normal_(self.modality.weight)
        self.prefix_adapter = PrefixInteractionAdapter(width)

    def hidden(self, input_ids, attention_mask, prefix_lengths):
        if input_ids.min() < 0 or input_ids.max() >= self.visual_start + self.visual_codes:
            raise ValueError('Token outside unified vocabulary')
        modality = (input_ids >= self.visual_start).long()
        embeddings = self.llm.get_input_embeddings()(input_ids) + self.modality(modality)
        embeddings = self.prefix_adapter(embeddings, modality, attention_mask.bool(),
                                         prefix_lengths, 'detection')
        return self.llm.model(inputs_embeds=embeddings, attention_mask=attention_mask,
                              use_cache=False, return_dict=True).last_hidden_state

    def forward(self, input_ids, attention_mask, prefix_lengths, labels):
        prefix = torch.arange(labels.shape[1], device=labels.device)[None] < prefix_lengths[:, None]
        if (labels[prefix] != -100).any() or (labels[~attention_mask.bool()] != -100).any():
            raise ValueError('Input/padding tokens must not be supervised')
        h = self.hidden(input_ids, attention_mask, prefix_lengths)
        target = labels[:, 1:]
        valid = target != -100
        h, target = h[:, :-1][valid], target[valid]
        if not len(target):
            raise ValueError('At least one supervised output token is required')
        weight = self.llm.get_output_embeddings().weight

        def projected_loss(hidden, matrix, truth):
            return F.cross_entropy(F.linear(hidden, matrix).float(), truth, reduction='sum')

        # Recompute vocabulary projections in backward instead of retaining
        # sequence_length x 136448 logits for every training example.
        total = h.new_zeros((), dtype=torch.float32)
        for start in range(0, len(target), self.loss_chunk):
            chunk = (h[start:start+self.loss_chunk], weight, target[start:start+self.loss_chunk])
            total = total + checkpoint(projected_loss, *chunk, use_reentrant=False)
        return {'loss_sum': total, 'target_tokens': valid.sum()}
