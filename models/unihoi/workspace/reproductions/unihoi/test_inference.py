import json
import math

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .detection import DetectionLanguageModel
from .evaluate_detection import EmptySafeNumpy
from .infer_detection import check_cached_forward, greedy_generate, materialize_rope
from .prediction import action_roles, native_records, parse_prediction, restore_box

GEOM = {"original_wh": [640, 480], "padded_side": 640, "pad_left_top": [0, 80]}
ACTIONS = {"cut": ["obj", "instr"], "stand": [], "point": ["instr"]}


def event():
    return {"human": [0, 125, 500, 750], "action": "cut", "roles": {
        "obj": {"box": [100, 250, 500, 500], "noun_id": 1}, "instr": None}}


def test_coordinate_inverse_inclusive_and_padding_rejection():
    assert restore_box([0, 125, 1000, 875], GEOM) == [0., 0., 639., 479.]
    np.testing.assert_allclose(restore_box([100, 250, 500, 500], GEOM), [64, 80, 319, 239])
    with pytest.raises(ValueError):
        restore_box([0, 0, 100, 100], GEOM)


def test_explicit_null_and_unpredicted_actions_not_fabricated():
    events, errors = parse_prediction(json.dumps([event()]), GEOM, 5, ACTIONS, [-.2, -.4])
    assert not errors and events[0]["roles"]["instr"] is None
    rows = native_records(events, ACTIONS)
    assert rows[0]["cut_instr"][:4] == [0.] * 4
    assert rows[0]["cut_instr"][4] == pytest.approx(math.exp(-.3))
    assert math.isnan(rows[0]["point_instr"][-1])
    assert math.isnan(rows[0]["stand_agent"])


@pytest.mark.parametrize("bad", ["missing", "noun", "box", "action"])
def test_invalid_event_retained_as_error_not_guessed(bad):
    value = event()
    if bad == "missing":
        del value["roles"]["instr"]
    elif bad == "noun":
        value["roles"]["obj"]["noun_id"] = 80
    elif bad == "box":
        value["human"] = [5, 5, 0, 0]
    else:
        value["action"] = "invented"
    events, errors = parse_prediction(json.dumps([value]), GEOM, 1, ACTIONS, [-.2])
    assert events == [] and errors


@pytest.mark.parametrize("text", ["prose", "{}", '[{"action":"cut","action":"stand"}]', '[NaN]'])
def test_invalid_json_or_duplicate_keys(text):
    events, errors = parse_prediction(text, GEOM, 1, ACTIONS, [-.2])
    assert events == [] and errors


def test_valid_empty_answer_and_multilabel_events():
    events, errors = parse_prediction("[]", GEOM, 1, ACTIONS, [-.2])
    assert events == errors == []
    second = {"human": event()["human"], "action": "stand", "roles": {}}
    events, errors = parse_prediction(json.dumps([event(), second]), GEOM, 1, ACTIONS, [-.2])
    assert len(events) == 2 and not errors
    assert action_roles(["cut_obj", "cut_instr", "stand_agent"]) == {"cut": ["obj", "instr"], "stand": []}


def tiny_model(meta=False):
    config = LlamaConfig(vocab_size=72, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "sdpa"
    if meta:
        with torch.device("meta"):
            return DetectionLanguageModel(LlamaForCausalLM(config), 64, 8)
    torch.manual_seed(42)
    return DetectionLanguageModel(LlamaForCausalLM(config), 64, 8).eval()


def test_cached_generation_equals_teacher_forcing_with_modality():
    model = tiny_model()
    prefix = torch.tensor([[1, 7, 65, 68, 9], [1, 8, 65, 67, 9]])
    check = check_cached_forward(model, prefix, torch.tensor([[5, 6], [7, 8]]))
    assert check["max_abs_hidden_difference"] < 1e-5
    assert check["next_token_argmax_equal"]
    generated, scores, finished = greedy_generate(model, prefix, -1, 3)
    seq = prefix
    expected = [[], []]
    with torch.no_grad():
        for _ in range(3):
            h = model.hidden(seq, torch.ones_like(seq), torch.tensor([5, 5]))
            logits = model.llm.lm_head(h[:, -1]).float()
            selected = logits.argmax(-1)
            for i, value in enumerate(selected.tolist()):
                expected[i].append(value)
            seq = torch.cat((seq, selected[:, None]), dim=1)
    assert generated == expected and finished == [False, False]
    assert all(len(s) == 3 and all(math.isfinite(v) and v <= 0 for v in s) for s in scores)


def test_eos_ends_each_answer():
    model = tiny_model()
    with torch.no_grad():
        model.llm.lm_head.weight.zero_()
    generated, scores, finished = greedy_generate(model, torch.tensor([[1, 7, 65, 68, 9]]), 0, 5)
    assert generated == [[0]] and finished == [True] and len(scores[0]) == 1


def test_meta_rope_buffers_rebuilt_using_installed_implementation():
    model = tiny_model(meta=True)
    materialize_rope(model)
    assert all(not b.is_meta for b in model.buffers())


def test_official_empty_class_guard_does_not_modify_numpy():
    wrapper = EmptySafeNumpy()
    assert wrapper.amax(np.array([])) == 0
    assert wrapper.empty_max_calls == 1
    assert wrapper.amax(np.array([.2, .7])) == np.amax(np.array([.2, .7]))
    assert wrapper.maximum is np.maximum
    with pytest.raises(ValueError):
        np.amax(np.array([]))
