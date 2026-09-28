"""MAP decoding of grounded participant--role sets, one best assignment per count state.

The functions below are identical to evaluation/native.py on the main branch, which decodes the CoRISP set
distribution. Pair-output baselines are decoded with the same functions on the independent product of their edge
scores (see evaluate_heir_sets.py --construction map), so every method uses one decoder.
"""
import numpy as np

MAX_EVENT_SETS = 8
MAX_IMAGE_SETS = 100


def _validate(weights, energy):
    weights = np.asarray(weights, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] < 1:
        raise ValueError('Log weights must have shape [candidates, 1 + active roles]')
    candidates, states = weights.shape
    if states > 7:
        raise ValueError('This implementation supports at most six active roles')
    if energy.shape != (candidates + 1, 3 ** (states - 1)):
        raise ValueError('Composition energy has an incompatible count-state shape')
    if np.isnan(weights).any() or np.isposinf(weights).any() or not np.isfinite(energy).all():
        raise ValueError('Weights allow finite values and -inf; potentials must be finite')
    return weights, energy


def _transitions(roles):
    states = 3 ** roles
    powers = 3 ** np.arange(roles)
    codes = np.arange(states)
    digits = (codes[:, None] // powers[None]) % 3
    return states, [
        (np.where(digits[:, role] >= 1, codes - powers[role], 0),
         digits[:, role] >= 1, codes, digits[:, role] == 2)
        for role in range(roles)
    ]


def log_partition_event(weights, energy):
    """Sum unary mass over assignments, then apply the terminal count potentials."""
    weights, energy = _validate(weights, energy)
    candidates, roles = weights.shape[0], weights.shape[1] - 1
    states, transitions = _transitions(roles)
    dp = np.full((candidates + 1, states), -np.inf)
    dp[0, 0] = 0.0
    for unary in weights:
        updated = dp + unary[0]
        for role, (first, valid_first, saturated, valid_saturated) in enumerate(transitions):
            first_mass = np.where(valid_first[None], dp[:-1, first], -np.inf)
            saturated_mass = np.where(valid_saturated[None], dp[:-1, saturated], -np.inf)
            selected = np.logaddexp(first_mass, saturated_mass) + unary[role + 1]
            updated[1:] = np.logaddexp(updated[1:], selected)
        dp = updated
    return float(np.logaddexp.reduce((dp + energy).ravel()))


def decode_state_winners(weights, energy, log_partition, max_hyp=MAX_EVENT_SETS):
    """Return (log probability, [(candidate, active-role index)]) per winning state.

    One assignment is retained per nonempty total-count/saturated-role-count
    state. Scores are normalized by the partition of the full distribution.
    Equal scores use count-state order, followed by the DP's fixed state order.
    """
    weights, energy = _validate(weights, energy)
    if not np.isfinite(log_partition):
        raise ValueError('The event partition must be finite')
    if not isinstance(max_hyp, int) or not 1 <= max_hyp <= MAX_EVENT_SETS:
        raise ValueError('max_hyp must be an integer between 1 and 8')
    candidates, roles = weights.shape[0], weights.shape[1] - 1
    states, transitions = _transitions(roles)
    dp = np.full((candidates + 1, states), -np.inf)
    dp[0, 0] = 0.0
    shape = (candidates, candidates + 1, states)
    back_count = np.zeros(shape, dtype=np.int32)
    back_state = np.zeros(shape, dtype=np.int32)
    back_role = np.zeros(shape, dtype=np.int8)
    counts = np.arange(candidates + 1)[:, None]
    codes = np.arange(states)[None, :]
    for index, unary in enumerate(weights):
        updated = dp + unary[0]
        previous_count = np.broadcast_to(counts, updated.shape).copy()
        previous_state = np.broadcast_to(codes, updated.shape).copy()
        chosen_role = np.full(updated.shape, -1, dtype=np.int8)
        for role, (first, valid_first, saturated, valid_saturated) in enumerate(transitions):
            first_score = np.where(valid_first[None], dp[:-1, first], -np.inf)
            saturated_score = np.where(valid_saturated[None], dp[:-1, saturated], -np.inf)
            use_saturated = saturated_score > first_score
            selected = np.where(use_saturated, saturated_score, first_score) + unary[role + 1]
            source = np.where(use_saturated, saturated[None], first[None])
            better = selected > updated[1:]
            updated[1:] = np.where(better, selected, updated[1:])
            previous_count[1:] = np.where(better, np.arange(candidates)[:, None], previous_count[1:])
            previous_state[1:] = np.where(better, source, previous_state[1:])
            chosen_role[1:] = np.where(better, role, chosen_role[1:])
        dp = updated
        back_count[index], back_state[index], back_role[index] = previous_count, previous_state, chosen_role
    terminal = dp + energy
    result = []
    for initial_count in range(1, candidates + 1):
        for initial_state in range(states):
            if not np.isfinite(terminal[initial_count, initial_state]):
                continue
            count, state, members = initial_count, initial_state, []
            for index in range(candidates - 1, -1, -1):
                role = int(back_role[index, count, state])
                if role >= 0:
                    members.append((index, role))
                count, state = int(back_count[index, count, state]), int(back_state[index, count, state])
            result.append((float(terminal[initial_count, initial_state] - log_partition), members[::-1]))
    result.sort(key=lambda item: -item[0])
    return result[:max_hyp]
