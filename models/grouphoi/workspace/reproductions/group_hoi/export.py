"""Official-cache conversion using the CURRENT Group_HOI forward contract.

The released generate_vcoco_official.py constructs an obsolete GEN_VLKT model.
Only its postprocessor/record semantics are reused here, not that model class.
Callers must strictly load an audited GroupHOI checkpoint and feature adapter.
"""

import math

import torch

VERBS = ('hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj',
         'hit_instr', 'hit_obj', 'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr',
         'talk_on_phone_instr', 'carry_obj', 'throw_obj', 'catch_obj', 'cut_instr',
         'cut_obj', 'run', 'work_on_computer_instr', 'ski_instr', 'surf_instr',
         'skateboard_instr', 'smile', 'drink_instr', 'kick_obj', 'point_instr',
         'read_obj', 'snowboard_instr')


def cache_records(result, image_id, missing_category_id=80):
    records = []
    for hoi in result['hoi_prediction']:
        subject = result['predictions'][hoi['subject_id']]
        entity = result['predictions'][hoi['object_id']]
        person_box = [float(x) for x in subject['bbox']]
        if len(person_box) != 4 or not all(math.isfinite(x) for x in person_box):
            raise ValueError('Invalid person box')
        object_box = ([math.nan] * 4 if int(entity['category_id']) == missing_category_id
                      else [float(x) for x in entity['bbox']])
        if int(entity['category_id']) != missing_category_id and (
                len(object_box) != 4 or not all(math.isfinite(x) for x in object_box)):
            raise ValueError('Invalid visible entity box')
        row = {'image_id': int(image_id), 'person_box': person_box}
        for name in VERBS:
            agent = name.rsplit('_', 1)[0] if '_' in name else name
            row[agent + '_agent'] = 0.
            if '_' in name:
                row[name] = [0., 0., .1, .1, 0.]
        if len(hoi['category_id']) != len(hoi['score']):
            raise ValueError('Misaligned actions and scores')
        for index, value in zip(hoi['category_id'], hoi['score']):
            index, score = int(index), float(value)
            # The author exporter adds sigmoid(HOI) and sigmoid(object)^2.
            # These are ranking scores, not calibrated probabilities.
            if not 0 <= index < len(VERBS) or not math.isfinite(score) or score < 0:
                raise ValueError('Invalid action index or confidence')
            name = VERBS[index]
            if '_' not in name:
                row[name + '_agent'] = max(row[name + '_agent'], score)
            else:
                row[name] = object_box + [score]
                agent = name.rsplit('_', 1)[0] + '_agent'
                row[agent] = max(row[agent], score)
        records.append(row)
    return records


@torch.no_grad()
def generate(model, vlm, postprocessor, loader, device, expected_ids, missing_category_id=80):
    model.eval()
    vlm.eval()
    records, coverage, empty_ids = [], [], []
    expected_ids = list(map(int, expected_ids))
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError('Expected IDs must be unique')
    for samples, targets in loader:
        samples = samples.to(device)
        clip_input = torch.stack([t['clip_inputs'] for t in targets]).to(device)
        outputs = model(samples, vlm, is_training=False, clip_input=clip_input)
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        results = postprocessor(outputs, sizes)
        if len(results) != len(targets):
            raise ValueError('One prediction result is required for every image')
        for result, target in zip(results, targets):
            image_id = int(target['img_id'])
            rows = cache_records(result, image_id, missing_category_id)
            coverage.append(image_id)
            records.extend(rows)
            if not rows:
                empty_ids.append(image_id)
    if len(coverage) != len(set(coverage)) or set(coverage) != set(expected_ids):
        raise ValueError('Incomplete or duplicated image coverage')
    return records, {'image_ids': sorted(coverage), 'images': len(coverage),
                     'empty_image_ids': sorted(empty_ids)}
