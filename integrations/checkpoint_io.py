"""Original checkpoint helpers without importing unrelated HICO models."""
from pathlib import Path
import hashlib
import torch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(checkpoint):
    state = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state, dict):
        raise TypeError('Checkpoint does not contain a model state dictionary.')
    return state


def _relocate_optimizer_state(optimizer, rank):
    device = torch.device('cuda', rank)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
