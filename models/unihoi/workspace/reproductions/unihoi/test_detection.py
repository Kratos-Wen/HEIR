import json

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .detection import DetectionLanguageModel, serialize_events


def tiny():
    torch.manual_seed(42)
    llm = LlamaForCausalLM(LlamaConfig(vocab_size=40, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64, attention_dropout=0.0))
    return DetectionLanguageModel(llm, visual_start=32, visual_codes=8, loss_chunk=2)


def batch():
    ids = torch.tensor([[1, 3, 32, 33, 34, 4, 5, 6, 2]])
    labels = ids.clone()
    labels[:, :6] = -100
    return dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids), prefix_lengths=torch.tensor([6]))


def test_chunked_ce_matches_full_ce_and_backpropagates_all_branches():
    model, b = tiny(), batch()
    h = model.hidden(b['input_ids'], b['attention_mask'], b['prefix_lengths'])
    full = torch.nn.functional.cross_entropy(model.llm.lm_head(h[:, :-1]).float().reshape(-1, 40),
        b['labels'][:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
    result = model(**b)
    torch.testing.assert_close(result['loss_sum'], full)
    assert int(result['target_tokens']) == 3
    result['loss_sum'].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert model.llm.model.embed_tokens.weight.grad[32:35].abs().sum() > 0
    assert model.prefix_adapter.iaa.q.weight.grad.abs().sum() > 0


def test_answer_suffix_cannot_change_prefix_or_earlier_logits():
    model, b = tiny().eval(), batch()
    h = model.hidden(b['input_ids'], b['attention_mask'], b['prefix_lengths'])
    altered = b['input_ids'].clone()
    altered[:, 7:] = 9
    h2 = model.hidden(altered, b['attention_mask'], b['prefix_lengths'])
    torch.testing.assert_close(h[:, :7], h2[:, :7])


def test_input_labels_are_rejected():
    model, b = tiny(), batch()
    b['labels'][0, 0] = 1
    with pytest.raises(ValueError, match='Input/padding'):
        model(**b)


def test_original_roles_unknowns_point_agent_only_and_letterbox():
    channels = ['cut_obj', 'cut_instr', 'point_instr', 'walk_agent', 'hold_obj', 'look_obj']
    p = dict(annotation_id=1, box=[0, 0, 100, 50], labels=[1, 1, 1, 1, -1, 0],
        visible=[True, True, False, False, False, False], objects=[2, 3, -1, -1, -1, -1],
        role_boxes=[[10, 10, 20, 20], [30, 10, 40, 20]] + [[0, 0, 0, 0]] * 4)
    events = json.loads(serialize_events({'people': [p]}, channels,
        {'padded_side': 100, 'pad_left_top': [0, 25]}))
    assert [e['action'] for e in events] == ['cut', 'point', 'walk']
    assert events[0]['human'] == [0, 250, 1000, 750]
    assert set(events[0]['roles']) == {'obj', 'instr'}
    assert events[1]['roles'] == {'instr': None}
    assert events[2]['roles'] == {}
    assert serialize_events({'people': []}, channels, {'padded_side': 100, 'pad_left_top': [0, 0]}) == '[]'
