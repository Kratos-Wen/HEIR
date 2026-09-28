import contextlib
import importlib.util
import io
import json
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from reproductions.incom_net.model import InCoMConfig, InCoMHead
from reproductions.incom_net.run import DATA
from .data import CHANNELS, NUM_ROLES, ROLE_NAMES, build_annotations, crop_targets
from .model import PaperMFTHead, CompleteDetector, candidate_pairs, associate, export_vcoco
from .evaluate import audited_evaluator
from .run import ensure_milestone


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def cfg():
    return InCoMConfig(detector_dim=8, vlm_dim=12, cnn_dim=10, hidden_dim=16,
                      heads=4, ffn_dim=32, dropout=0, num_actions=len(CHANNELS))


def record():
    return {'boxes': torch.tensor([[0., 0., 10., 10.], [10., 10., 20., 20.]]),
            'normalized_boxes': torch.tensor([[0., 0., .5, .5], [.5, .5, 1., 1.]]),
            'labels': torch.tensor([0, 1]), 'scores': torch.tensor([.9, .8]),
            'detector_layers': torch.randn(3, 2, 8), 'vlm_layers': torch.randn(3, 4, 12),
            'cnn_tokens': torch.randn(4, 10), 'grid': (2, 2), 'size': torch.tensor([20, 20])}


def target():
    labels = torch.zeros(1, len(CHANNELS), dtype=torch.long)
    labels[0, [0, 6, 7, 24, 25]] = 1
    visible = torch.zeros_like(labels, dtype=torch.bool)
    visible[0, [0, 6]] = True
    boxes = torch.zeros(1, len(CHANNELS), 4)
    boxes[visible] = torch.tensor([10., 10., 20., 20.])
    objects = torch.full_like(labels, -1)
    objects[visible] = 1
    return {'boxes': torch.tensor([[0., 0., 10., 10.]]), 'labels': labels, 'visible': visible,
            'role_boxes': boxes, 'objects': objects, 'person_ids': torch.tensor([100]), 'image_id': 1}


def compatibility():
    out = torch.ones(80, len(CHANNELS), dtype=torch.bool)
    out[:, NUM_ROLES:] = False
    return out


def original_fixture():
    actions = {}
    for channel in CHANNELS:
        name, role = channel.rsplit('_', 1)
        actions.setdefault(name, ['agent'])
        if role != 'agent':
            actions[name].append(role)
    vsrl = []
    for name, roles in actions.items():
        role_ids = [100, 101]
        for role in roles[1:]:
            role_ids.extend([102 if name == 'hold' else 0, 0])
        vsrl.append({'action_name': name, 'role_name': roles, 'ann_id': [100, 101],
                     'image_id': [1, 2], 'label': [1, 1], 'role_object_id': role_ids})
    coco = {'images': [{'id': i, 'file_name': f'{i}.jpg'} for i in (1, 2)],
            'categories': [{'id': 1}, {'id': 5}], 'annotations': [
                {'id': 100, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]},
                {'id': 101, 'image_id': 2, 'category_id': 1, 'bbox': [1, 2, 10, 10]},
                {'id': 102, 'image_id': 1, 'category_id': 5, 'bbox': [10, 10, 10, 10]},
                {'id': 103, 'image_id': 1, 'category_id': 1, 'bbox': [25, 25, 10, 10]}]}
    return vsrl, coco


def test_original_labels_preserve_null_point_agent_and_unknown_people():
    data = build_annotations(*original_fixture(), [1, 2])
    assert data['counts']['visible'] == 1
    assert data['counts']['null'] == 49
    assert data['counts']['agent'] == 8
    assert data['counts']['null:point_instr'] == 2
    unknown = data['annotations'][0]['people'][1]
    assert unknown['labels'] == [-1] * len(CHANNELS)
    assert sum(sum(x) for x in data['compatibility']) == 1
    assert data['compatibility'][1][0]


def test_ingestion_rejects_out_of_split_labels():
    vsrl, coco = original_fixture()
    with pytest.raises(ValueError, match='out-of-split'):
        build_annotations(vsrl, coco, [1])


def test_resume_milestone_recovery_is_idempotent_and_never_overwrites(tmp_path):
    latest, milestone = tmp_path / 'latest.pth', tmp_path / 'epoch30.pth'
    latest.write_bytes(b'fixed-checkpoint')
    ensure_milestone(latest, milestone)
    ensure_milestone(latest, milestone)
    assert milestone.read_bytes() == b'fixed-checkpoint'
    different = tmp_path / 'different.pth'
    different.write_bytes(b'other-checkpoint')
    with pytest.raises(ValueError, match='different milestone'):
        ensure_milestone(different, milestone)


def test_mft_masks_before_affine_not_after_and_keeps_full_forward():
    new = PaperMFTHead(cfg()).eval()
    old = InCoMHead(cfg()).eval()
    old.load_state_dict(new.state_dict(), strict=True)
    r = record()
    pairs = candidate_pairs(r['labels'])[:1]
    context = torch.randn(2, cfg().hidden_dim)
    args = (r['detector_layers'][-1], context, r['cnn_tokens'], r['vlm_layers'][-1], pairs)
    torch.testing.assert_close(new.reason(*args, 'full')['logits'], old.reason(*args, 'full')['logits'])
    seen, outputs, disabled = [], [], []
    hooks = [new.detector_pair.register_forward_pre_hook(lambda m, args: seen.append(args[0])),
             new.detector_pair.register_forward_hook(lambda m, args, result: outputs.append(result))]
    hooks += [x.detector_attn.register_forward_hook(lambda *a: disabled.append(1)) for x in new.decoder]
    new.reason(*args, 'vlm_only')
    for h in hooks:
        h.remove()
    assert torch.count_nonzero(seen[0]) == 0
    assert torch.count_nonzero(outputs[0]) > 0
    assert not disabled
    torch.testing.assert_close(outputs[0], new.detector_pair(torch.zeros_like(seen[0])))


def test_multi_role_targets_do_not_confuse_visible_with_missing():
    r, t = record(), target()
    pairs = candidate_pairs(r['labels'])
    assert pairs.tolist() == [[0, 1], [0, 2]]
    y, valid = associate(r, pairs, t, compatibility())
    assert y[0].nonzero().flatten().tolist() == [0, 6]
    assert y[1].nonzero().flatten().tolist() == [7, 24, 25]
    assert not valid[0, 25:].any()


def test_unknown_person_never_becomes_negative_supervision():
    r, t = record(), target()
    t['labels'].fill_(-1)
    y, valid = associate(r, candidate_pairs(r['labels']), t, compatibility())
    assert not y.any() and not valid.any()


def test_crop_cannot_create_a_missing_object_positive():
    t = crop_targets(target(), 0, 0, 9)
    assert t['labels'][0, 0] == -1
    assert t['labels'][0, 6] == -1
    assert t['labels'][0, 7] == 1
    r = record()
    r['boxes'][0] = torch.tensor([0., 0., 9., 9.])
    y, valid = associate(r, candidate_pairs(r['labels']), t, compatibility())
    assert not y[:, 0].any() and not valid[:, 0].any()
    assert y[1, 7] == 1


def test_all_people_get_a_distinct_null_pair():
    pairs = candidate_pairs(torch.tensor([0, 1, 0]))
    assert pairs[-2:].tolist() == [[0, 3], [2, 3]]
    assert not (pairs[:, 0] == pairs[:, 1]).any()


class Extractor(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, images):
        return [self.value for _ in images]


def test_complete_forward_backward_and_prediction_export():
    r = record()
    model = CompleteDetector(Extractor(r), cfg(), compatibility())
    result = model([torch.empty(0)], [target()])
    result['loss'].backward()
    assert result['matched_positive_edges'] == 5
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    model.eval()
    prediction = model([torch.empty(0)])[0]
    output = export_vcoco(prediction, 1, (40, 40))
    assert len(output) == 1 and output[0]['person_box'] == [0., 0., 20., 20.]
    assert all(name in output[0] for name in CHANNELS)
    assert all(name.rsplit('_', 1)[0] + '_agent' in output[0] for name in ROLE_NAMES)


@pytest.mark.parametrize('labels', [[1, 2], []])
def test_no_detected_people_produces_empty_differentiable_output(labels):
    r = record()
    n = len(labels)
    r['labels'] = torch.tensor(labels, dtype=torch.long)
    for k in ('boxes', 'normalized_boxes', 'scores'):
        r[k] = r[k][:n]
    r['detector_layers'] = r['detector_layers'][:, :n]
    model = CompleteDetector(Extractor(r), cfg(), compatibility())
    result = model([torch.empty(0)], [target()])
    assert result['loss'] == 0
    result['loss'].backward()
    model.eval()
    assert export_vcoco(model([torch.empty(0)])[0], 1, (20, 20)) == []


def test_null_export_is_zero_box_with_learned_score():
    r = record()
    scores = torch.full((2, len(CHANNELS)), .1)
    scores[1, 7] = .9
    p = {'boxes': r['boxes'], 'pairs': candidate_pairs(r['labels']), 'scores': scores, 'size': r['size']}
    out = export_vcoco(p, 1, (20, 20))[0]
    assert out['eat_instr'][:4] == [0.] * 4
    assert out['eat_instr'][4] == pytest.approx(.9)
    assert out['eat_agent'] == pytest.approx(.9)


def official_class():
    spec = importlib.util.spec_from_file_location('incom_v2_test_evaluator', DATA / 'vsrl_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VCOCOeval


def test_collector_checks_every_image_and_empty_case():
    base = official_class()
    stats = {'images': 0, 'empty_images': 0, 'empty_case_checked': False}
    cls = audited_evaluator(base, stats)
    instance = cls.__new__(cls)
    instance.image_ids = [1, 2]
    instance.actions, instance.roles, instance.num_actions = ['hold'], [['agent', 'obj']], 1
    instance._audit_records = {1: [{'image_id': 1, 'person_box': [0, 0, 10, 10],
                                  'hold_agent': .9, 'hold_obj': [0, 0, 0, 0, .9]}]}
    original = instance._collect_detections_for_image
    instance._collect_detections_for_image = lambda records, i: original(instance._audit_records.get(i, []), i)
    assert stats == {'images': 2, 'empty_images': 1, 'empty_case_checked': True}


def test_collector_rejects_mismatched_cache():
    base = official_class()
    cls = audited_evaluator(base, {'images': 0, 'empty_images': 0})
    instance = cls.__new__(cls)
    instance.image_ids, instance._audit_records = [1], {}
    instance.actions, instance.roles, instance.num_actions = ['hold'], [['agent', 'obj']], 1
    with pytest.raises(ValueError, match='collector mismatch'):
        instance._collect_detections_for_image = lambda records, i: (np.ones((1, 5)), np.ones((1, 5, 2)))


@pytest.mark.parametrize('box,s1,s2', [([0., 0., 0., 0.], 100., 100.), ([10., 10., 20., 20.], 0., 100.)])
def test_missing_role_convention_against_actual_official_evaluator(tmp_path, monkeypatch, box, s1, s2):
    monkeypatch.setattr(np, 'bool', np.bool_, raising=False)
    base = official_class()
    evaluator = base.__new__(base)
    evaluator.actions, evaluator.roles, evaluator.num_actions = ['point', 'hold', 'eat'], [['agent', 'instr'], ['agent', 'obj'], ['agent', 'obj', 'instr']], 3
    row = {'image_id': 1, 'person_box': [0., 0., 10., 10.]}
    for action, roles in zip(evaluator.actions, evaluator.roles):
        for role in roles:
            row[action + '_' + role] = .9 if role == 'agent' else box + [.9]
    path = tmp_path / 'predictions.pkl'
    with path.open('wb') as handle:
        pickle.dump([row], handle)
    gt = [{'id': 1, 'boxes': np.array([[0., 0., 10., 10.]], dtype=np.float32),
           'gt_classes': np.array([1]), 'gt_actions': np.ones((1, 3)), 'gt_role_id': np.full((1, 3, 2), -1)}]
    for scenario, expected in [('scenario_1', s1), ('scenario_2', s2)]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            evaluator._do_role_eval(gt, str(path), eval_type=scenario)
        assert f'Average Role [{scenario}] AP = {expected:.2f}' in output.getvalue()
