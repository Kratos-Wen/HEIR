import math

import torch

from .diffusion import ForwardDiffusion, ground_truth, initialize
from .transition import exact_transition
from .vcoco import pair_targets, regression_loss, select_pairs, positive_coverage
from .native_eval import records_from_scores
from .detector import COCO_IDS


def fixture():
    channels = ['cut_obj', 'cut_instr', 'point_instr', 'stand_agent', 'hold_obj']
    person = {'annotation_id': 7, 'box': [0., 0., 50., 100.], 'labels': [1, 1, 1, 1, -1],
        'visible': [True, True, False, False, False], 'objects': [4, 7, -1, -1, -1],
        'role_boxes': [[50., 50., 100., 100.], [50., 0., 100., 50.], [0.]*4, [0.]*4, [0.]*4]}
    cache = {'human_boxes': torch.tensor([[0., 0., .5, 1.]] * 3),
        'entity_boxes': torch.tensor([[.5, .5, 1., 1.], [.5, 0., 1., .5], [0.]*4]),
        'human_query': torch.tensor([0, 0, 0]), 'entity_query': torch.tensor([1, 2, -1]),
        'predicted_noun': torch.tensor([2, 3, 80]), 'null': torch.tensor([False, False, True])}
    return channels, {'people': [person], 'image_id': 12}, cache


def test_roles_null_unknown_and_noun_targets():
    channels, row, cache = fixture()
    labels, nouns = pair_targets(cache, row, 100, 100, channels)
    assert labels.tolist() == [[1, 0, 0, -1, -1], [0, 1, 0, -1, -1], [0, 0, 1, 1, -1]]
    assert nouns.tolist() == [4, 7, 80]
    assert positive_coverage(cache, row, 100, 100, labels)['covered_positive_edges'] == 4
    cache['human_boxes'] += 2
    labels, _ = pair_targets(cache, row, 100, 100, channels)
    assert (labels == -1).all()


def test_sampling_coverage_rotation_and_no_unknown_negatives():
    labels = torch.cat((torch.ones(12, 3), torch.zeros(20, 3), torch.full((2, 3), -1)))
    samples = [select_pairs(labels, e, 99) for e in range(30)]
    assert all(len(x) == 8 and (x < 32).all() for x in samples)
    assert set(range(12)) <= set(torch.cat(samples).tolist())
    assert not len(select_pairs(torch.full((2, 3), -1), 0, 99))


def test_exact_sampled_trajectory_matches_recurrence_at_first_step():
    process = ForwardDiffusion(steps=3, trials=20)
    clean = ground_truth(torch.tensor([0, 1]), torch.tensor([[1., 0.], [0., 1.]]), 2)
    prior = initialize(torch.tensor([[.6, .4], [.3, .7]]), 2)
    torch.manual_seed(42)
    current, previous = exact_transition(clean, prior, torch.ones(2, dtype=torch.long), process)
    torch.testing.assert_close(previous, clean)
    torch.manual_seed(42)
    torch.testing.assert_close(current, process.step(clean, prior, 1))


def test_unknown_slices_have_zero_loss_and_gradient():
    prediction = torch.randn(2, 3, 4, 2, requires_grad=True)
    target = torch.randn_like(prediction)
    known = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
    value, count = regression_loss(prediction, target, known)
    value.backward()
    assert count == 3
    assert not prediction.grad.permute(0, 2, 1, 3)[~known].count_nonzero()


def test_native_export_keeps_distinct_roles_and_virtual_missing_object():
    channels, row, cache = fixture()
    scores = torch.tensor([[.9, .1, .1, .9, .1], [.1, .8, .1, .9, .2], [.1, .1, .7, .6, .3]])
    records = records_from_scores(scores, cache, channels, 12, 100, 100)
    assert len(records) == 1
    r = records[0]
    assert r['cut_obj'][:4] == [50., 50., 100., 100.]
    assert r['cut_instr'][:4] == [50., 0., 100., 50.]
    assert all(math.isnan(x) for x in r['point_instr'][:4])
    assert abs(r['stand_agent'] - .6) < 1e-6
    assert abs(r['cut_agent'] - .9) < 1e-6


def test_sparse_coco_classifier_ids():
    assert len(COCO_IDS) == len(set(COCO_IDS)) == 80
    assert COCO_IDS[0] == 1 and COCO_IDS[-1] == 90 and 12 not in COCO_IDS


def test_train_image_path_respects_coco_source_split():
    from .run import train_image_path
    for split in ('train2014', 'val2014'):
        name = f'COCO_{split}_000000000165.jpg'
        path = train_image_path(name)
        assert path.parent.name == split and path.name == name
