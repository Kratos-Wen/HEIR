import numpy as np
import pytest

from evaluation.export import native_prediction
from evaluation.native import log_partition_event, decode_state_winners


def prediction(action_count=1):
    weights = np.array([[0., 2.], [0., 3.]])
    energy = np.zeros((3, 3))
    return dict(boxes=np.array([[1., 2., 3., 4.]] * 3), labels=np.array([0, 1, 1]),
                entity_scores=np.ones(3), pairs=np.array([[0, 1], [0, 2]]), size=[10, 20],
                joint_scores=np.ones((2, action_count, 2)), agent_entities=np.array([0]),
                events=[dict(agent_index=0, action_index=a, pair_indices=np.array([0, 1]),
                             active_role_indices=np.array([1]), log_state_weights=weights,
                             composition_energy=energy, log_partition=log_partition_event(weights, energy))
                        for a in range(action_count)])


def test_export_uses_shared_ids_original_boxes_and_active_roles():
    result = native_prediction(prediction(), 'image', [20, 40], ['person', 'object'], ['hold'], ['T', 'I'])
    assert result['entities'][0]['box'] == [2., 4., 6., 8.]
    assert result['sets'][0]['members'] == [{'entity_id': 1, 'role': 'I'}, {'entity_id': 2, 'role': 'I'}]
    assert result['sets'][0]['subject_id'] == 0
    assert len(result['sets']) == 2


def test_image_budget_applies_after_event_decoding():
    result = native_prediction(prediction(60), 'image', [20, 40], ['person', 'object'],
                               [f'a{a}' for a in range(60)], ['T', 'I'])
    assert len(result['sets']) == 100
    scores = [item['score'] for item in result['sets']]
    assert scores == sorted(scores, reverse=True)


def test_invalid_subject_identity_is_rejected():
    output = prediction()
    output['pairs'][1, 0] = 2
    with pytest.raises(ValueError, match='another subject'):
        native_prediction(output, 'image', [20, 40], ['person', 'object'], ['hold'], ['T', 'I'])


def test_empty_event_produces_no_nonempty_sets():
    weights = np.empty((0, 2))
    energy = np.zeros((1, 3))
    assert decode_state_winners(weights, energy, log_partition_event(weights, energy)) == []
