"""Compare exact coefficients, losses, gradients and optimizer updates."""
import argparse
import json
import time
import torch

from . import environment
from corisp.role_arity_event_field import RoleArityEventField as Field
from .batched_dp import partition


def check(device, roles, candidates, dtype, masked=False):
    torch.manual_seed(120+roles+candidates)
    a = torch.randn(3, candidates, 1+roles, dtype=dtype, device=device).requires_grad_()
    energy = torch.randn(3, candidates+1, 3**roles, dtype=dtype, device=device).requires_grad_()
    b = a.detach().clone().requires_grad_()
    other = energy.detach().clone().requires_grad_()
    if masked and candidates:
        mask = torch.rand_like(a) < .2
        mask[..., 0] = False
    else:
        mask = torch.zeros_like(a, dtype=torch.bool)
    aa, bb = a.masked_fill(mask, -torch.inf), b.masked_fill(mask, -torch.inf)
    active = torch.arange(roles, device=device)
    ref = [Field.log_arity_partition(w, active, e) for w, e in zip(aa, energy)]
    z, c = partition(Field, bb, other)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(c, torch.stack([x[1] for x in ref]), atol=tolerance, rtol=tolerance)
    target = torch.stack([x[0] for x in ref])
    torch.testing.assert_close(z, target, atol=tolerance, rtol=tolerance)
    # Non-uniform upstream weights exercise independent events, not only sum(z).
    scale = z.new_tensor([.3, 1., 2.7])
    (target*scale).sum().backward()
    (z*scale).sum().backward()
    grad_error = 0.
    for x, y in ((a, b), (energy, other)):
        if x.grad is None and y.grad is None:
            continue
        torch.testing.assert_close(x.grad, y.grad, atol=tolerance, rtol=tolerance)
        grad_error = max(grad_error, float((x.grad-y.grad).abs().max()))
    op1 = torch.optim.AdamW([a, energy], lr=1e-4)
    op2 = torch.optim.AdamW([b, other], lr=1e-4)
    op1.step(); op2.step()
    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(energy, other, atol=tolerance, rtol=tolerance)
    return {'roles': roles, 'candidates': candidates, 'dtype': str(dtype),
            'masked': masked, 'max_gradient_error': grad_error}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--output')
    args = p.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    rows = [check(args.device, r, n, d, masked) for r, n in ((1, 0), (2, 4), (6, 8), (6, 29))
            for d in (torch.float32, torch.float64) for masked in (False, True)]
    result = {'passed': True, 'device': args.device, 'cases': rows, 'seconds': time.monotonic()-start}
    print(json.dumps(result), flush=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
