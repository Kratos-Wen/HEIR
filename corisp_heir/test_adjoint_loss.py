"""Structured targets, selective recomputation and multi-step optimizer parity."""
import copy
import io
import os
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from . import environment
from .batched_execution import losses
from corisp import RoleArityEventField, RoleArityEventFieldConfig, RoleFillerTargetGroup


class AdjointLossTests(unittest.TestCase):
    def test_checkpoint_execution_overrides_shell_and_old_default(self):
        with patch.dict(os.environ, HEIR_CORISP_DP_BACKEND='adjoint', HEIR_CORISP_METADATA_CACHE='1'):
            environment.restore_execution({})
            self.assertEqual(os.environ['HEIR_CORISP_DP_BACKEND'], 'autograd')
            self.assertEqual(os.environ['HEIR_CORISP_METADATA_CACHE'], '0')
            environment.restore_execution(dict(dp_backend='adjoint', compiled_dp=True,
                metadata_cache=True, selective_recomputation=True))
            self.assertEqual(os.environ['HEIR_CORISP_DP_BACKEND'], 'adjoint')
            self.assertEqual(os.environ['HEIR_CORISP_SELECTIVE_RECOMPUTE'], '1')
            with self.assertRaises(ValueError):
                environment.restore_execution(dict(dp_backend='approximate'))

    def test_low_precision_energy_boundary(self):
        from .adjoint_dp import partition
        from .test_adjoint_dp import reference
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(256)
            w = torch.randn(2, 4, 7, requires_grad=True)
            e = torch.randn(2, 5, 729, dtype=dtype, requires_grad=True)
            v, f = w.detach().clone().requires_grad_(), e.detach().clone().requires_grad_()
            a, _ = reference(w, e)
            b, _ = partition(v, f)
            a.sum().backward(); b.sum().backward()
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(w.grad, v.grad, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(e.grad, f.grad, atol=0, rtol=0)

    def test_set_likelihood_updates_and_restore(self):
        torch.set_num_threads(1)
        torch.manual_seed(426)
        field = RoleArityEventField(RoleArityEventFieldConfig(d_model=8, num_roles=6,
            max_cardinality=6, num_heads=2, arity_rank=4, dropout=0.)).eval()
        with torch.no_grad():
            field.arity_log_scale.fill_(.3)
            field.cardinality_head.weight.normal_(std=.05)
        other = copy.deepcopy(field)
        logits = torch.randn(2, 4, 7, requires_grad=True)
        clone = logits.detach().clone().requires_grad_()
        params = [list(field.parameters())+[logits], list(other.parameters())+[clone]]
        optimizers = [torch.optim.AdamW(p, lr=1e-4) for p in params]
        states, contexts = torch.randn(2, 8), torch.randn(2, 6, 8)
        groups = [RoleFillerTargetGroup(0, torch.tensor([0, 1])),
                  RoleFillerTargetGroup(0, torch.tensor([0, 1])),
                  RoleFillerTargetGroup(4, torch.tensor([2, 3]))]
        descriptions = [(0, 0, torch.arange(4), groups), (0, 1, torch.arange(4), [])]
        for step in range(3):
            values = []
            for backend, model, raw, optimizer in zip(('autograd', 'adjoint'),
                    (field, other), (logits, clone), optimizers):
                optimizer.zero_grad(set_to_none=True)
                visible = raw.softmax(-1)[..., 1:]
                with patch.dict(os.environ, HEIR_CORISP_DP_BACKEND=backend, HEIR_CORISP_COMPILE_DP='0'):
                    if backend == 'autograd':
                        value = checkpoint(lambda v: losses(model, v, states, contexts, descriptions, .5, .1),
                                           visible, use_reentrant=False).sum()
                    else:
                        value = losses(model, visible, states, contexts, descriptions, .5, .1,
                                       recompute_potentials=True).sum()
                    value.backward()
                values.append(value.detach())
            torch.testing.assert_close(*values, atol=3e-6, rtol=3e-6)
            for a, b in zip(*params):
                self.assertEqual(a.grad is None, b.grad is None)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)
            for optimizer, parameters in zip(optimizers, params):
                torch.nn.utils.clip_grad_norm_(parameters, .1, error_if_nonfinite=True)
                optimizer.step()
            for a, b in zip(*params):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            if step == 1:
                buffer = io.BytesIO()
                torch.save(dict(model=other.state_dict(), optimizer=optimizers[1].state_dict()), buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
                other.load_state_dict(saved['model'], strict=True)
                optimizers[1].load_state_dict(saved['optimizer'])

    def test_marginal_normalization(self):
        from .adjoint_dp import partition
        torch.manual_seed(78)
        w = torch.randn(2, 8, 7, dtype=torch.float64, requires_grad=True)
        e = torch.randn(2, 9, 729, dtype=torch.float64, requires_grad=True)
        z, _ = partition(w, e)
        dw, de = torch.autograd.grad(z.sum(), (w, e))
        torch.testing.assert_close(dw.sum(-1), torch.ones(2, 8, dtype=w.dtype))
        torch.testing.assert_close(de.flatten(1).sum(-1), torch.ones(2, dtype=e.dtype))
        self.assertTrue(bool((dw >= 0).all() and (de >= 0).all()))


if __name__ == '__main__':
    unittest.main()
