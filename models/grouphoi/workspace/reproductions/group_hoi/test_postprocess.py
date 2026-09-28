from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .export import cache_records
from .postprocess import GroupCachePostprocessor


@pytest.mark.parametrize('batch,queries', [(1, 2), (2, 3), (2, 2)])
@pytest.mark.parametrize('nms', [False, True])
def test_actual_author_postprocessor_with_native_layout(batch, queries, nms):
    args = SimpleNamespace(num_queries=queries, subject_category_id=0,
                           use_nms_filter=nms, thres_nms=.7, nms_alpha=1., nms_beta=.5)
    processor = GroupCachePostprocessor(args, np.ones((29, 80), dtype=np.float32))
    classes = len(processor.author.vcoco_triplet_labels)
    outputs = {'pred_hoi_logits': torch.full((queries, batch, classes), 4.),
               'pred_obj_logits': torch.full((batch, queries, 82), -4.),
               'pred_sub_boxes': torch.tensor([.5, .5, .4, .4]).expand(batch, queries, 4),
               'pred_obj_boxes': torch.tensor([.3, .3, .2, .2]).expand(batch, queries, 4)}
    outputs['pred_obj_logits'][:, :, 80] = 4.
    sizes = torch.tensor([[100, 200]]).repeat(batch, 1)
    results = processor(outputs, sizes)
    assert len(results) == batch
    for i, result in enumerate(results):
        rows = cache_records(result, i)
        assert len(rows) == (1 if nms else queries)
        assert all(row['hold_obj'][-1] > 1 for row in rows)
        assert all(np.isnan(row['hold_obj'][:4]).all() for row in rows)
        np.testing.assert_allclose(rows[0]['person_box'], [60, 30, 140, 70], atol=1e-5)
    assert outputs['pred_hoi_logits'].shape == (queries, batch, classes)


def test_wrong_layout_fails_instead_of_silently_mixing_images():
    args = SimpleNamespace(num_queries=3, subject_category_id=0,
                           use_nms_filter=False, thres_nms=.7, nms_alpha=1., nms_beta=.5)
    processor = GroupCachePostprocessor(args, np.ones((29, 80)))
    with pytest.raises(ValueError, match='QBC'):
        processor({'pred_hoi_logits': torch.zeros(2, 3, 263),
                   'pred_obj_logits': torch.zeros(2, 3, 82)}, torch.ones(2, 2))
