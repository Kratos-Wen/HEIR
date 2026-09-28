import torch
import pytest

from .vcoco import crop_pair_targets, associate_pairs, export_vcoco, ACTION_NAMES, InCoMDetector
from .model import InCoMConfig


@pytest.mark.parametrize('width,height', [(1333, 333), (100, 1333), (1, 800), (800, 800)])
def test_panorama_crop_always_fits_after_long_side_cap(width, height):
    from .vcoco import square_crop_region
    for _ in range(30):
        top, left, side = square_crop_region(width, height)
        assert 0 < side <= min(width, height, 600)
        assert top >= 0 and left >= 0
        assert top + side <= height and left + side <= width


def test_crop_checks_each_box_against_its_own_origin():
    target = {'boxes_h': torch.tensor([[0., 0., 2., 2.], [8., 8., 12., 12.]]),
              'boxes_o': torch.tensor([[6., 6., 8., 8.], [1., 1., 2., 2.]]),
              'actions': torch.tensor([0, 1]), 'objects': torch.tensor([1, 2])}
    result = crop_pair_targets(target, 0, 0, 10, 10)
    # First human lies left/above its object but remains a valid independent box.
    assert result['actions'].tolist() == [0, 1]
    result = crop_pair_targets(target, 0, 0, 4, 4)
    assert len(result['actions']) == 0
    assert target['boxes_h'][1, 2] == 12


def test_same_pair_multiple_roles_and_wrong_noun():
    record = {'boxes': torch.tensor([[0., 0., 10., 10.], [12., 12., 20., 20.], [12., 12., 20., 20.]]),
              'labels': torch.tensor([0, 1, 2])}
    target = {'boxes_h': record['boxes'][[0, 0]], 'boxes_o': record['boxes'][[1, 1]],
              'actions': torch.tensor([4, 5]), 'objects': torch.tensor([1, 1])}
    pairs = torch.tensor([[0, 1], [0, 2]])
    labels, valid = associate_pairs(record, pairs, target, torch.ones(80, 24, dtype=torch.bool))
    assert labels[0, 4] == labels[0, 5] == 1
    assert labels.sum() == 2 and labels[1].sum() == 0
    assert valid.shape == (2, 24)


def test_native_export_includes_both_role_types_without_coordinate_drift():
    p = {'boxes': torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
         'pairing': torch.tensor([[0, 1], [0, 1]]), 'scores': torch.tensor([.8, .7]),
         'labels': torch.tensor([ACTION_NAMES.index('cut_instr'), ACTION_NAMES.index('cut_obj')]),
         'size': torch.tensor([10, 20])}
    rows = export_vcoco(p, 77, (40, 20))
    assert rows[0]['person_box'] == [2., 4., 6., 8.]
    assert rows[0]['cut_instr'][:4] == [10., 12., 14., 16.]
    assert rows[1]['cut_obj'][:4] == [10., 12., 14., 16.]
    assert rows[0]['cut_obj'][-1] == 0 and rows[1]['cut_instr'][-1] == 0
    assert rows[0]['point_instr'][-1] == 0
    assert len([k for k in rows[0] if k.endswith('_agent')]) == 26


def test_supervised_detector_wrapper_optimizer_and_inference():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = InCoMConfig(detector_dim=8, vlm_dim=12, cnn_dim=10, hidden_dim=16,
                      heads=4, ffn_dim=32, dropout=0, num_actions=24)
    record = {'boxes': torch.tensor([[0., 0., 4., 8.], [4., 0., 8., 8.]]),
              'labels': torch.tensor([0, 1]), 'scores': torch.tensor([.9, .8]),
              'size': torch.tensor([8, 8]), 'detector_layers': torch.randn(3, 2, 8),
              'vlm_layers': torch.randn(3, 4, 12), 'cnn_tokens': torch.randn(6, 10),
              'normalized_boxes': torch.tensor([[0., 0., .5, 1.], [.5, 0., 1., 1.]]),
              'grid': (2, 2)}

    class FixedFeatures(torch.nn.Module):
        def forward(self, images):
            return [record for _ in images]

    model = InCoMDetector(FixedFeatures(), cfg, torch.ones(80, 24, dtype=torch.bool))
    target = {'boxes_h': record['boxes'][[0]], 'boxes_o': record['boxes'][[1]],
              'actions': torch.tensor([4]), 'objects': torch.tensor([1])}
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    before = model.head.classifier.weight.detach().clone()
    out = model([torch.zeros(3, 8, 8)], [target])
    assert out['matched_positive_edges'] == 1
    torch.testing.assert_close(out['loss'].detach(), out['mft_full'] + out['mft_detector_only'] + out['mft_vlm_only'])
    out['loss'].backward()
    optimizer.step()
    assert not torch.equal(before, model.head.classifier.weight)
    model.eval()
    predictions = model([torch.zeros(3, 8, 8)])
    assert len(predictions[0]['labels']) == 24
    assert predictions[0]['pair_features'].shape == (1, 16)
    assert len(export_vcoco(predictions[0], 5, (8, 8))) == 24


def test_training_function_never_reads_test_labels_or_selects_test_checkpoints():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(__file__).with_name('run.py').read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    calls = [node.func.id for node in ast.walk(train) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert 'RGBTest' not in calls and 'test' not in calls
    assert not any(isinstance(node, ast.Constant) and isinstance(node.value, str) and 'vcoco_test' in node.value
                   for node in ast.walk(train))
