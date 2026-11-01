import math

import pytest
import torch

from .export import cache_records, generate


def result(missing=False):
    return {'predictions': [{'bbox': [1, 2, 31, 52], 'category_id': 0},
                            {'bbox': [10, 20, 40, 50], 'category_id': 80 if missing else 1}],
            'hoi_prediction': [{'subject_id': 0, 'object_id': 1,
                                'category_id': [16, 17, 26], 'score': [.4, .8, .2]}]}


def test_multiple_roles_and_point_exported_without_changing_boxes():
    row, = cache_records(result(), 17)
    assert row['cut_instr'] == [10., 20., 40., 50., .4]
    assert row['cut_obj'][-1] == row['cut_agent'] == .8
    assert row['point_instr'][-1] == .2
    assert len([k for k in row if k.endswith('_agent')]) == 26


def test_true_missing_entity_keeps_official_nan_convention():
    row, = cache_records(result(True), 17)
    assert all(math.isnan(x) for x in row['cut_obj'][:4])
    assert row['cut_obj'][-1] == .8


def test_author_ranking_scores_above_one_are_not_clipped():
    output = result()
    output['hoi_prediction'][0]['score'] = [1.4, 1.8, 1.2]
    row, = cache_records(output, 17)
    assert row['cut_obj'][-1] == row['cut_agent'] == 1.8


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1])
def test_invalid_ranking_scores_rejected(score):
    output = result()
    output['hoi_prediction'][0]['score'][0] = score
    with pytest.raises(ValueError, match='confidence'):
        cache_records(output, 17)


def test_current_group_model_signature_and_empty_image_coverage():
    class Model(torch.nn.Module):
        def forward(self, samples, vlm, *, is_training, clip_input):
            assert not is_training and clip_input.shape == (2, 3, 8, 8)
            return None
    targets = [{'img_id': i, 'orig_size': torch.tensor([10, 20]),
                'clip_inputs': torch.zeros(3, 8, 8)} for i in (17, 19)]
    loader = [(torch.zeros(2, 3, 8, 8), targets)]
    processor = lambda out, sizes: [result(), {'predictions': [], 'hoi_prediction': []}]
    rows, coverage = generate(Model(), torch.nn.Identity(), processor, loader, 'cpu', [17, 19])
    assert len(rows) == 1 and coverage['empty_image_ids'] == [19]
    with pytest.raises(ValueError, match='coverage'):
        generate(Model(), torch.nn.Identity(), processor, loader, 'cpu', [17, 20])
