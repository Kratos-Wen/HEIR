"""Independent enumeration and frozen-recurrence checks of exact HEIR DP."""
import argparse
from itertools import product
import json
from pathlib import Path
import time

import torch

from . import environment
from corisp.role_arity_event_field import RoleArityEventField as Field
from .adjoint_dp import partition


def reference(weights, energy):
    active = torch.arange(weights.shape[-1]-1, device=weights.device)
    results = [Field.log_arity_partition(w, active, e) for w, e in zip(weights, energy)]
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


def check(device, roles, candidates, dtype, mode, compiled=False):
    torch.manual_seed(421+roles+candidates)
    a = torch.randn(2, candidates, 1+roles, dtype=dtype, device=device).requires_grad_()
    e = torch.randn(2, candidates+1, 3**roles, dtype=dtype, device=device).requires_grad_()
    b, f = a.detach().clone().requires_grad_(), e.detach().clone().requires_grad_()
    mask = torch.zeros_like(a, dtype=torch.bool)
    if mode == 'masked':
        mask = torch.rand_like(a) < .3
        mask[..., 0] = False
    elif mode == 'null_only':
        mask[..., 1:] = True
    aa, bb = a.masked_fill(mask, -torch.inf), b.masked_fill(mask, -torch.inf)
    z0, c0 = reference(aa, e)
    z1, c1 = partition(bb, f, compiled=compiled)
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
    error = 0.
    for x, y in ((a, b), (e, f)):
        if x.grad is None:
            assert y.grad is None or not y.grad.count_nonzero()
            continue
        torch.testing.assert_close(x.grad, y.grad, atol=tol, rtol=tol)
        if x.numel():
            error = max(error, float((x.grad-y.grad).abs().max()))
    if candidates:
        o0 = torch.optim.AdamW([a, e], lr=1e-4)
        o1 = torch.optim.AdamW([b, f], lr=1e-4)
        o0.step(); o1.step()
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
        torch.testing.assert_close(e, f, atol=tol, rtol=tol)
    return dict(roles=roles, candidates=candidates, dtype=str(dtype), mode=mode,
                max_gradient_error=error)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--compiled', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    core = environment.verify_core()
    rows = []
    for roles, candidates in ((1, 0), (2, 4), (6, 4), (6, 8), (6, 29)):
        for dtype in (torch.float32, torch.float64):
            for mode in ('all', 'masked', 'null_only'):
                row = check(args.device, roles, candidates, dtype, mode, args.compiled)
                rows.append(row)
                print(json.dumps(row), flush=True)
    torch.manual_seed(42)
    w = torch.randn(1, 3, 3, dtype=torch.float64, device=args.device).requires_grad_()
    e = torch.randn(1, 4, 9, dtype=torch.float64, device=args.device).requires_grad_()
    assert torch.autograd.gradcheck(lambda w, e: partition(w, e, compiled=args.compiled)[0], (w, e))
    assert environment.verify_core() == core
    report = dict(passed=True, cases=rows, device=args.device, compiled=args.compiled,
                  gradcheck=True, frozen_core_files=len(core), seconds=time.monotonic()-started,
                  scope='first derivatives and exact marginals; no higher-order derivative support')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
