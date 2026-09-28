"""Exact support conditioning: energy, likelihood, gradient and role-ID parity."""
import copy
import os
import unittest
from unittest.mock import patch

import torch

from . import environment
from .batched_execution import losses, potentials
from .batched_dp import partition
from .support import state_indices
from corisp import RoleArityEventField, RoleArityEventFieldConfig, RoleFillerTargetGroup


class SupportTests(unittest.TestCase):
    def test_release_person_first_mapping(self):
        import json
        from heir_protocol.compatibility import Compatibility
        from heir_training.data import noun_order
        from .support import aligned_support
        from heir_protocol.closed import digest, encoded
        document = dict(schema='heir_semantic_compatibility_v1', status='HUMAN_APPROVED',
                        null_state_allowed=True,
                        vocabulary=dict(verbs=['hold', 'sit'], nouns=['chair', 'person'],
                                        roles=['target', 'support']),
                        vocabulary_sha256='synthetic', role_bits_verb_noun=[[1, 1], [2, 0]])
        document['content_sha256'] = digest(encoded(document))
        table = Compatibility(document)
        axes = dict(table.axes, nouns=['person', 'chair'])
        mask, active = aligned_support(table, axes)
        self.assertEqual(axes['nouns'][0], 'person')
        self.assertEqual(int(mask.sum()), 3)
        for ni, noun in enumerate(axes['nouns']):
            for vi, verb in enumerate(axes['verbs']):
                expected = [table.allows(verb,noun,r) for r in axes['roles']]
                self.assertEqual(mask[ni,vi].tolist(),expected)
        self.assertTrue(all(active))
        with self.assertRaises(ValueError):
            aligned_support(table,dict(axes,nouns=axes['nouns'][:-1]))

    def test_dense_compact_loss_gradient(self):
        torch.set_num_threads(1)
        for active in ((5,), (1,4), (0,2,5), tuple(range(6))):
            for backend in ('autograd', 'adjoint'):
                torch.manual_seed(433)
                field = RoleArityEventField(RoleArityEventFieldConfig(d_model=8,num_roles=6,
                    max_cardinality=8,num_heads=2,arity_rank=4,dropout=0.)).eval()
                with torch.no_grad():
                    field.arity_log_scale.fill_(.4)
                    field.cardinality_head.weight.normal_(std=.05)
                other = copy.deepcopy(field)
                raw = torch.randn(2,4,7,requires_grad=True)
                clone = raw.detach().clone().requires_grad_()
                e,c = torch.randn(2,8),torch.randn(2,6,8)
                mask = torch.zeros(2,4,6,dtype=torch.bool)
                mask[:,:,list(active)] = True
                mask[:,3,:] = False
                groups = [RoleFillerTargetGroup(active[-1],torch.tensor([0,1])),
                          RoleFillerTargetGroup(active[-1],torch.tensor([0,1]))]
                desc = [(0,0,torch.arange(4),groups),(0,1,torch.arange(4),[])]
                with patch.dict(os.environ,HEIR_CORISP_DP_BACKEND=backend,HEIR_CORISP_COMPILE_DP='0',HEIR_CORISP_METADATA_CACHE='1'):
                    dense = losses(field,raw.softmax(-1)[...,1:],e,c,desc,.5,.1,
                                   active_roles=tuple(range(6)),support=mask)
                    compact = losses(other,clone.softmax(-1)[...,1:],e,c,desc,.5,.1,
                                     recompute_potentials=True,active_roles=active,support=mask)
                    torch.testing.assert_close(dense,compact,atol=3e-6,rtol=3e-6)
                    dense.sum().backward(); compact.sum().backward()
                    torch.testing.assert_close(raw.grad,clone.grad,atol=4e-6,rtol=4e-5)
                    for a,b in zip(field.parameters(),other.parameters()):
                        self.assertEqual(a.grad is None,b.grad is None)
                        if a.grad is not None:
                            self.assertTrue(torch.isfinite(b.grad).all())
                            torch.testing.assert_close(a.grad,b.grad,atol=4e-6,rtol=4e-5)

    def test_energy_is_full_submatrix_and_marginals(self):
        torch.manual_seed(13)
        f = RoleArityEventField(RoleArityEventFieldConfig(d_model=8,num_roles=6,
            max_cardinality=8,num_heads=2,arity_rank=4,dropout=0.)).eval()
        with torch.no_grad():
            f.arity_log_scale.fill_(.5)
        active=(1,3,5)
        mask=torch.zeros(1,3,6,dtype=torch.bool)
        mask[:,:,list(active)]=True
        v=torch.randn(1,3,7).softmax(-1)[...,1:]
        e,c=torch.randn(1,8),torch.randn(1,6,8)
        with patch.dict(os.environ,HEIR_CORISP_DP_BACKEND='adjoint',HEIR_CORISP_COMPILE_DP='0',HEIR_CORISP_METADATA_CACHE='1'):
            wd,ed=potentials(f,v,e,c,tuple(range(6)),mask)
            wc,ec=potentials(f,v,e,c,active,mask)
            _,codes=state_indices(active,6,e.device)
            torch.testing.assert_close(ec,ed[:,:,codes],atol=2e-6,rtol=2e-6)
            zd,_=partition(f,wd.requires_grad_(),ed)
            zc,_=partition(f,wc.requires_grad_(),ec)
            gd=torch.autograd.grad(zd.sum(),wd)[0]
            gc=torch.autograd.grad(zc.sum(),wc)[0]
            torch.testing.assert_close(gd[:,:,[0,2,4,6]],gc,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(gc.sum(-1),torch.ones(1,3),atol=2e-6,rtol=2e-6)
            self.assertEqual(float(gd[:,:,[1,3,5]].abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
