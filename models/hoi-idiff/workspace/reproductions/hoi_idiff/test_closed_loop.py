import torch
from torch import nn

from .closed_loop import trajectory, backward_rollout
from .diffusion import ForwardDiffusion, initialize, ground_truth


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([.1, -.1]))
        self.inputs = []

    def forward(self, state, steps, appearance):
        self.inputs.append((state.detach().clone(), state.requires_grad, steps.clone()))
        return self.logits.softmax(-1).expand_as(state)


def inputs():
    clean = ground_truth(torch.zeros(2, dtype=torch.long), torch.tensor([[1.], [0.]]), 1)
    prior = initialize(torch.ones(2, 1), 1)
    process = ForwardDiffusion(steps=3, trials=10)
    return clean, prior, process


def test_full_forward_trajectory_matches_eq4():
    clean, prior, process = inputs()
    torch.manual_seed(42)
    states = trajectory(clean, prior, process)
    torch.manual_seed(42)
    expected = clean
    assert len(states) == 4
    for k in range(1, 4):
        expected = process.step(expected, prior, k)
        torch.testing.assert_close(states[k], expected)


def test_reverse_uses_own_outputs_and_inference_boundary():
    clean, prior, process = inputs()
    states = trajectory(clean, prior, process)
    model = RecordingModel()
    known = torch.ones(2, 1, dtype=torch.bool)
    total, last = backward_rollout(model, states, prior, torch.zeros(2, 3), known,
                                  known.sum(), samples_per_pair=2)
    assert [int(v[2][0]) for v in model.inputs] == [3, 2, 1]
    torch.testing.assert_close(model.inputs[0][0][0], prior[0])
    torch.testing.assert_close(model.inputs[0][0][1], states[-1][1])
    for state, requires_grad, _ in model.inputs[1:]:
        torch.testing.assert_close(state, last)
        assert not requires_grad
    assert total > 0 and torch.isfinite(model.logits.grad).all()


def test_empty_rank_is_graph_connected_zero():
    clean, prior, process = inputs()
    model = RecordingModel()
    total, _ = backward_rollout(model, trajectory(clean, prior, process), prior,
        torch.zeros(2, 3), torch.zeros(2, 1, dtype=torch.bool), torch.tensor(10),
        world_size=8, samples_per_pair=2)
    assert total == 0
    assert model.logits.grad is not None and not model.logits.grad.count_nonzero()


def test_global_normalization_matches_combined_batch():
    clean, prior, process = inputs()
    states = trajectory(clean, prior, process)
    known = torch.ones(2, 1, dtype=torch.bool)
    full = RecordingModel()
    # M=1 gives identical inference-boundary handling in split and full cases.
    backward_rollout(full, states, prior, torch.zeros(2, 3), known, known.sum(), samples_per_pair=1)
    gradients = []
    for rank in range(2):
        local = RecordingModel()
        backward_rollout(local, [s[rank:rank+1] for s in states], prior[rank:rank+1],
            torch.zeros(1, 3), known[rank:rank+1], known.sum(), world_size=2, samples_per_pair=1)
        gradients.append(local.logits.grad)
    torch.testing.assert_close(torch.stack(gradients).mean(0), full.logits.grad)
