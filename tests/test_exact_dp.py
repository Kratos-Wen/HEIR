"""Exact HEIR DP against enumeration and the per-event recurrence."""
from itertools import product

import pytest
import torch

from corisp_heir.exact_dp import partition
from corisp import EventSetField


def reference(weights, energy):
    active = torch.arange(weights.shape[-1]-1, device=weights.device)
    results = [EventSetField.log_arity_partition(w, active, e) for w, e in zip(weights, energy)]
    return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])


def enumeration(weights, energy):
    batch, candidates, channels = weights.shape
    masses = []
    for assignment in product(range(channels), repeat=candidates):
        counts = [assignment.count(r) for r in range(1, channels)]
        arity = sum(min(c, 2)*3**r for r, c in enumerate(counts))
        mass = energy[:, sum(counts), arity]
        if candidates:
            mass = mass + weights[:, torch.arange(candidates), list(assignment)].sum(-1)
        masses.append(mass)
    return torch.logsumexp(torch.stack(masses), dim=0)


@pytest.mark.parametrize('roles,candidates', [(1, 0), (2, 4), (6, 4), (6, 8)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('mode', ['all', 'masked', 'null_only'])
def test_partition_coefficients_gradients_and_updates(roles, candidates, dtype, mode):
    torch.manual_seed(421+roles+candidates)
    a = torch.randn(2, candidates, 1+roles, dtype=dtype).requires_grad_()
    e = torch.randn(2, candidates+1, 3**roles, dtype=dtype).requires_grad_()
    b, f = a.detach().clone().requires_grad_(), e.detach().clone().requires_grad_()
    mask = torch.zeros_like(a, dtype=torch.bool)
    if mode == 'masked':
        mask = torch.rand_like(a) < .3
        mask[..., 0] = False
    elif mode == 'null_only':
        mask[..., 1:] = True
    aa, bb = a.masked_fill(mask, -torch.inf), b.masked_fill(mask, -torch.inf)
    z0, c0 = reference(aa, e)
    z1, c1 = partition(bb, f, compiled=False)
    tol = 3e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(c0, c1, atol=tol, rtol=tol)
    torch.testing.assert_close(z0, z1, atol=tol, rtol=tol)
    if candidates <= 4:
        torch.testing.assert_close(z1, enumeration(bb, f), atol=tol, rtol=tol)
    scale = a.new_tensor([.7, -1.3])
    # Exercise the coefficient-output VJP too, with signed independent weights.
    upstream = torch.randn_like(c0)
    finite = torch.isfinite(c0)
    ((z0*scale).sum()+(c0.masked_fill(~finite, 0)*upstream).sum()*.03).backward()
    ((z1*scale).sum()+(c1.masked_fill(~finite, 0)*upstream).sum()*.03).backward()
    for x, y in ((a, b), (e, f)):
        if x.grad is None:
            assert y.grad is None or not y.grad.count_nonzero()
            continue
        torch.testing.assert_close(x.grad, y.grad, atol=tol, rtol=tol)
    if candidates:
        first, second = torch.optim.AdamW([a, e], lr=1e-4), torch.optim.AdamW([b, f], lr=1e-4)
        first.step()
        second.step()
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
        torch.testing.assert_close(e, f, atol=tol, rtol=tol)


def test_analytic_gradient_passes_gradcheck():
    torch.manual_seed(42)
    w = torch.randn(1, 3, 3, dtype=torch.float64).requires_grad_()
    e = torch.randn(1, 4, 9, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda w, e: partition(w, e, compiled=False)[0], (w, e))


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_low_precision_energy_boundary(dtype):
    torch.manual_seed(256)
    w = torch.randn(2, 4, 7, requires_grad=True)
    e = torch.randn(2, 5, 729, dtype=dtype, requires_grad=True)
    v, f = w.detach().clone().requires_grad_(), e.detach().clone().requires_grad_()
    a, _ = reference(w, e)
    b, _ = partition(v, f)
    a.sum().backward()
    b.sum().backward()
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(w.grad, v.grad, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(e.grad, f.grad, atol=0, rtol=0)


def test_marginal_normalization():
    torch.manual_seed(78)
    w = torch.randn(2, 8, 7, dtype=torch.float64, requires_grad=True)
    e = torch.randn(2, 9, 729, dtype=torch.float64, requires_grad=True)
    z, _ = partition(w, e)
    dw, de = torch.autograd.grad(z.sum(), (w, e))
    torch.testing.assert_close(dw.sum(-1), torch.ones(2, 8, dtype=w.dtype))
    torch.testing.assert_close(de.flatten(1).sum(-1), torch.ones(2, dtype=e.dtype))
    assert bool((dw >= 0).all() and (de >= 0).all())
