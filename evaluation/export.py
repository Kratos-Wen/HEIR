"""Export native HEIR sets with shared image-local proposal identities."""
import math
import numpy as np

from .native import MAX_IMAGE_SETS, decode_state_winners


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def native_prediction(output, image_id, original_size, nouns, actions, roles):
    boxes = array(output['boxes']).astype(float, copy=True)
    height, width = array(output['size']).reshape(2)
    original_height, original_width = array(original_size).reshape(2)
    if min(height, width, original_height, original_width) <= 0:
        raise ValueError('Image dimensions must be positive')
    boxes *= [original_width / width, original_height / height] * 2
    labels = array(output['labels']).astype(int)
    scores = array(output['entity_scores'])
    if boxes.shape != (len(labels), 4) or scores.shape != labels.shape or not np.isfinite(boxes).all():
        raise ValueError('Invalid proposal arrays')
    entities = [{'id': j, 'noun': nouns[n], 'box': box.tolist(), 'score': float(score)}
                for j, (box, n, score) in enumerate(zip(boxes, labels, scores))]
    pairs = array(output['pairs']).astype(int).reshape(-1, 2)
    agents = array(output['agent_entities']).astype(int)
    hypotheses = []
    for event in output['events']:
        subject = int(agents[event['agent_index']])
        action = actions[event['action_index']]
        pair_rows = array(event['pair_indices']).astype(int)
        role_rows = array(event['active_role_indices']).astype(int)
        if not np.all(pairs[pair_rows, 0] == subject):
            raise ValueError('Event contains pairs from another subject')
        for log_probability, members in decode_state_winners(
                array(event['log_state_weights']), array(event['composition_energy']),
                float(array(event['log_partition']))):
            assignment = [{'entity_id': int(pairs[pair_rows[m], 1]), 'role': roles[role_rows[r]]}
                          for m, r in members]
            hypotheses.append({'subject_id': subject, 'action': action, 'members': assignment,
                               'score': math.exp(min(0., log_probability)),
                               'log_probability': log_probability})
    hypotheses.sort(key=lambda item: -item['log_probability'])
    return {'image_id': image_id, 'size': [int(original_height), int(original_width)],
            'entities': entities, 'sets': hypotheses[:MAX_IMAGE_SETS],
            'pairs': pairs.tolist(), 'role_scores': array(output['joint_scores']).tolist(),
            'actions': list(actions), 'roles': list(roles)}
