import itertools

import numpy as np
import pytest

from evaluation.native import decode_state_winners, log_partition_event


@pytest.mark.parametrize('roles,candidates', [(1, 4), (2, 4), (3, 3)])
def test_state_winners_against_enumeration(roles, candidates):
    rng = np.random.default_rng(15)
    weights = rng.normal(size=(candidates, roles + 1))
    weights[0, -1] = -np.inf
    energy = rng.normal(size=(candidates + 1, 3 ** roles))
    state_best = {}
    terms = []
    for assignment in itertools.product(range(roles + 1), repeat=candidates):
        count = sum(s != 0 for s in assignment)
        code = sum(min(2, assignment.count(r + 1)) * 3 ** r for r in range(roles))
        score = sum(weights[i, s] for i, s in enumerate(assignment)) + energy[count, code]
        terms.append(score)
        if count and np.isfinite(score):
            key = (count, code)
            members = [(i, s - 1) for i, s in enumerate(assignment) if s]
            if key not in state_best or score > state_best[key][0]:
                state_best[key] = (score, members)
    log_z = float(np.logaddexp.reduce(terms))
    assert log_partition_event(weights, energy) == pytest.approx(log_z, abs=1e-10)
    expected = sorted(state_best.values(), key=lambda item: -item[0])[:8]
    actual = decode_state_winners(weights, energy, log_z)
    assert len(actual) == len(expected)
    for (score, members), (expected_score, expected_members) in zip(actual, expected):
        assert score == pytest.approx(expected_score - log_z, abs=1e-10)
        assert members == expected_members
    score, members = decode_state_winners(weights, energy, log_z, max_hyp=1)[0]
    assert score == pytest.approx(expected[0][0] - log_z, abs=1e-10)
    assert members == expected[0][1]
