import importlib.util
from pathlib import Path

import pytest
import torch

from .model import (InCoMConfig, InCoMHead, ContextAttention, box_patch_masks,
                    human_entity_pairs, focal_mft_loss, inference_scores)
from .backbones import FrozenCLIPLayers, official_modules, select_proposals, PVIC, build_detr_without_download


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(42)


@pytest.fixture
def case():
    cfg = InCoMConfig(detector_dim=8, vlm_dim=12, cnn_dim=10, hidden_dim=16,
                      heads=4, ffn_dim=32, dropout=0, num_actions=5, instance_chunk=2)
    q = torch.randn(3, 3, 8)
    v = torch.randn(3, 6, 12)
    b = torch.tensor([[0., 0., .5, 1.], [.3, .2, 1., .8], [.01, .01, .02, .02]])
    pairs = human_entity_pairs(torch.tensor([0, 1, 0]))
    return InCoMHead(cfg), (q, v, b, (2, 3), torch.randn(4, 10), pairs)


def test_masks_overlap_and_other_instances():
    b = torch.tensor([[0., 0., 1., 1.], [.5, .5, 1., 1.]])
    own, other = box_patch_masks(b, (2, 2))
    assert own[0].all()
    assert torch.equal(other[0], own[1])
    assert torch.equal(other[1], own[0])
    assert other[0, -1] and own[0, -1]


def test_single_instance_tiny_and_empty_masks():
    own, surrounding = box_patch_masks(torch.tensor([[.01, .01, .02, .02]]), (2, 2))
    assert own.sum() == 1 and not surrounding.any()
    a, b = box_patch_masks(torch.empty(0, 4), (2, 3))
    assert a.shape == b.shape == (0, 6)


@pytest.mark.parametrize("box", [[0., 0., 0., 1.], [-.1, 0., 1., 1.], [0., 0., float('nan'), 1.]])
def test_bad_boxes_fail(box):
    with pytest.raises(ValueError):
        box_patch_masks(torch.tensor([box]), (2, 2))


def test_pair_enumeration_all_people_not_only_first():
    pairs = human_entity_pairs(torch.tensor([2, 0, 3, 0]))
    assert pairs.tolist() == [[1, 0], [1, 2], [1, 3], [3, 0], [3, 1], [3, 2]]
    assert human_entity_pairs(torch.tensor([2, 3])).shape == (0, 2)


def test_icr_uses_multiplicative_kv_not_unmasked_features(case):
    head, _ = case
    module = ContextAttention(head.cfg)
    x = torch.randn(2, 6, 16)
    mask = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    inputs = []
    hook = module.attn.register_forward_pre_hook(lambda m, args: inputs.append(args))
    result = module(x, mask)
    hook.remove()
    torch.testing.assert_close(inputs[0][0], x)
    torch.testing.assert_close(inputs[0][1], x * mask[..., None])
    torch.testing.assert_close(inputs[0][2], x * mask[..., None])
    assert torch.equal(result[1], torch.zeros_like(result[1]))


def test_mft_shapes_finite_gradients_and_evaluation(case):
    head, inputs = case
    branches = head(*inputs)
    assert set(branches) == {"full", "detector_only", "vlm_only"}
    labels = torch.zeros(len(inputs[-1]), 5)
    labels[0, :2] = 1
    valid = torch.ones_like(labels, dtype=torch.bool)
    loss, terms = focal_mft_loss(branches, labels, valid, alpha=.5, gamma=.1)
    torch.testing.assert_close(loss, terms['full'] + terms['detector_only'] + terms['vlm_only'])
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    head.eval()
    assert set(head(*inputs)) == {"full"}


def test_mft_is_not_random_feature_dropout(case):
    head, inputs = case
    head.eval()
    q, v, b, grid, cnn, pairs = inputs
    context = head.mine_context(q, v, b, grid)
    disabled_calls = []
    hooks = [layer.vlm_attn.register_forward_hook(lambda *args: disabled_calls.append(1)) for layer in head.decoder]
    first = head.reason(q[-1], context, cnn, v[-1], pairs, 'detector_only')
    second = head.reason(q[-1], context * 50, cnn, v[-1] * -10, pairs, 'detector_only')
    assert not disabled_calls
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)
    first = head.reason(q[-1], context, cnn, v[-1], pairs, 'vlm_only')
    with torch.no_grad():
        head.detector_pair[0].bias.fill_(1234)
    second = head.reason(q[-1] * 20, context, cnn * 40, v[-1], pairs, 'vlm_only')
    torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)


def test_instance_permutation_equivariance(case):
    head, inputs = case
    head.eval()
    q, v, b, grid, cnn, pairs = inputs
    perm = torch.tensor([2, 0, 1])
    inverse = perm.argsort()
    original = head(*inputs)['full']['logits']
    changed = head(q[:, perm], v, b[perm], grid, cnn, inverse[pairs])['full']['logits']
    torch.testing.assert_close(original, changed, rtol=1e-5, atol=1e-6)


def test_chunking_preserves_the_algorithm(case):
    head, inputs = case
    head.eval()
    expected = head(*inputs)['full']['logits']
    for layer in head.context_levels:
        layer.chunk = 1
    torch.testing.assert_close(head(*inputs)['full']['logits'], expected, rtol=1e-5, atol=1e-6)


def test_no_pairs_gives_differentiable_empty_loss(case):
    head, inputs = case
    branches = head(*inputs[:-1], torch.empty(0, 2, dtype=torch.long))
    loss, _ = focal_mft_loss(branches, torch.empty(0, 5), torch.empty(0, 5, dtype=torch.bool), alpha=.5, gamma=.1)
    assert loss == 0
    loss.backward()


def test_invalid_labels_cannot_disappear(case):
    head, inputs = case
    branches = head(*inputs)
    labels = torch.zeros(len(inputs[-1]), 5)
    labels[0, 0] = 1
    with pytest.raises(ValueError, match="Positive annotation"):
        focal_mft_loss(branches, labels, torch.zeros_like(labels, dtype=torch.bool), alpha=.5, gamma=.1)
    with pytest.raises(ValueError, match="all three"):
        focal_mft_loss({"full": branches['full']}, labels, torch.ones_like(labels, dtype=torch.bool), alpha=.5, gamma=.1)


def test_focal_matches_reference_and_ignores_invalid_negatives():
    logits = torch.tensor([[.1, -2., 500.]])
    branches = {name: {"logits": logits} for name in ("full", "detector_only", "vlm_only")}
    labels = torch.tensor([[1., 0., 0.]])
    valid = torch.tensor([[True, True, False]])
    result, _ = focal_mft_loss(branches, labels, valid, alpha=.5, gamma=.1)
    from torchvision.ops import sigmoid_focal_loss
    reference = 3 * sigmoid_focal_loss(logits[:, :2], labels[:, :2], alpha=.5, gamma=.1, reduction="sum")
    torch.testing.assert_close(result, reference)


def test_inference_supplement_exponent():
    logits = torch.zeros(1, 3)
    scores = inference_scores(logits, torch.tensor([[.8, .5]]), torch.tensor([[1, 0, 1]], dtype=torch.bool))
    torch.testing.assert_close(scores, torch.tensor([[.5 * .4**2.8, 0, .5 * .4**2.8]]))


def test_clip_layer_extractor_matches_original_block_outputs():
    official = official_modules()
    visual = official.VisionTransformer(8, 4, 16, 4, 4, 8)
    extractor = FrozenCLIPLayers(visual, levels=3)
    seen = []
    hooks = [block.register_forward_hook(lambda m, args, output: seen.append(output.detach().clone())) for block in visual.transformer.resblocks]
    image = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        visual(image)
    expected = torch.stack([x[1:].permute(1, 0, 2) for x in seen[-3:]], dim=1)
    for hook in hooks:
        hook.remove()
    actual, grid = extractor(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert grid == (2, 2)
    extractor.train()
    assert not visual.training and not extractor.training
    assert all(not p.requires_grad for p in extractor.parameters())


def test_proposal_query_ids_match_native_pvic_selection():
    spec = importlib.util.spec_from_file_location("incom_test_pvic_ops", PVIC / "ops.py")
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    result = {"scores": torch.tensor([.9, .8, .01, .7, .2]),
              "labels": torch.tensor([0, 1, 2, 0, 1]),
              "boxes": torch.tensor([[0, 0, 3, 3], [1, 1, 4, 4], [2, 2, 5, 5], [0, 0, 3, 3], [6, 6, 7, 7]], dtype=torch.float)}
    hidden = torch.randn(1, 5, 256)
    sizes = torch.tensor([[8, 8]])
    native = ops.prepare_region_proposals([result], hidden, sizes, .05, 0, 3, 15)[0]
    ours = select_proposals(result, sizes[0])
    for key in ("boxes", "scores", "labels"):
        torch.testing.assert_close(ours[key], native[key])
    torch.testing.assert_close(hidden[0, ours['query_ids']], native['hidden_states'])


def test_paper_recipe_and_no_corisp_imports():
    import ast
    import json
    spec = json.loads(Path(__file__).with_name('reconstruction.json').read_text())
    p = spec['paper_specified']
    assert p['confidence_exponent'] == 2.8 and p['lr_drop_epochs'] == [10, 20]
    assert p['interaction_width'] == 384 and p['context_levels'] == 3
    for path in Path(__file__).parent.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('corisp')


def test_detr_constructor_preserves_official_shapes_without_download():
    model, _ = build_detr_without_download()
    assert model.input_proj.weight.shape == (256, 2048, 1, 1)
    assert model.class_embed.weight.shape == (81, 256)
    assert model.query_embed.weight.shape == (100, 256)
    assert len(model.transformer.decoder.layers) == 6
