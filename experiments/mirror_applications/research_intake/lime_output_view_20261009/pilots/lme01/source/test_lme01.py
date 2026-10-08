#!/usr/bin/env python3
"""Smoke tests for LME01; no heldout test metrics consumed."""
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_lme01 import SharedAdapterMultiOutput, METHODS, K, D, data_world, patterns

class Lme01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_test_disjoint(self):
        _,_,_,_,m=data_world(71)
        self.assertFalse(set(m["train_ids"]) & set(m["audit_ids"]))
        self.assertEqual(m["train_count"]+m["test_count"],1797)

    def test_five_outputs_and_one_heavy_call(self):
        x=torch.randn(7,64)
        for meth in METHODS:
            mdl=SharedAdapterMultiOutput(meth,71)
            c=[0,0]
            h1=mdl.trunk.register_forward_hook(lambda *args:c.__setitem__(0,c[0]+1))
            h2=mdl.adapter.register_forward_hook(lambda *args:c.__setitem__(1,c[1]+1))
            y=mdl(x)
            h1.remove();h2.remove()
            self.assertEqual(y.shape,(7,K,2))
            self.assertEqual(c,[1,1])

    def test_role_specific_codes_change_logits(self):
        x=torch.randn(9,64)
        for meth in ("lime_full_diagonal","lime_linear4","mirror_rot4","mirror_shear4"):
            mdl=SharedAdapterMultiOutput(meth,71)
            with torch.no_grad():
                v=getattr(mdl,"p",None) if meth=="lime_full_diagonal" else mdl.m
                v[0].fill_(0)
                v[1].fill_(0.45)
            y=mdl(x)
            self.assertGreater(float((y[:,0]-y[:,1]).abs().max()),1e-7)

    def test_native_givens_same_math(self):
        mdl=SharedAdapterMultiOutput("mirror_rot4",83)
        delta=torch.randn(11,D)
        with torch.no_grad():
            angles=mdl.m @ mdl.fixed_planes.T
            d=delta.reshape(11,D//2,2)
            native=[]
            for j in range(K):
                a,b=d[...,0],d[...,1]
                ca,sa=angles[j].cos(),angles[j].sin()
                native.append(torch.stack((ca*a-sa*b,sa*a+ca*b),-1).reshape(11,D))
            expected=torch.stack(native,dim=1)
            observed=mdl._member_readout(delta)
        self.assertLess(float((observed-expected).abs().max()),3e-7)

    def test_rotation_norm(self):
        mdl=SharedAdapterMultiOutput("mirror_rot4",83)
        d=torch.randn(10,D)
        views=mdl._member_readout(d)
        self.assertLess(float((views.norm(dim=-1)-d.norm(dim=-1,keepdim=True)).abs().max()),1e-5)

    def test_gradient_all_methods(self):
        for meth in METHODS:
            mdl=SharedAdapterMultiOutput(meth,71)
            x=torch.randn(6,64)
            y=mdl(x)
            loss=y.square().mean()
            loss.backward()
            self.assertIsNotNone(mdl.trunk[0].weight.grad)
            self.assertTrue(torch.isfinite(mdl.trunk[0].weight.grad).all())
            self.assertIsNotNone(mdl.adapter[0].weight.grad)
            self.assertTrue(torch.isfinite(mdl.adapter[0].weight.grad).all())

    def test_serialized_complete(self):
        x,y,_,_,meta=data_world(71)
        for meth in METHODS:
            mdl=SharedAdapterMultiOutput(meth,71)
            blob=mdl.stored(meta)
            self.assertGreater(len(blob),5000)
            with np.load(io.BytesIO(blob),allow_pickle=False) as data:
                self.assertIn("__meta__",data.files)
                self.assertIn("trunk.0.weight",data.files)
                self.assertIn("__preprocess_mean__",data.files)
                m=bytes(data["__meta__"]).decode()
                self.assertIn(meth,m)
                self.assertEqual(data["__preprocess_mean__"].shape,(64,))
                if meth=="native_onepass_linearheads":
                    self.assertNotIn("common.weight",data.files)

    def test_train_initialization_aligned(self):
        methods=[SharedAdapterMultiOutput(m,72) for m in METHODS]
        ref=methods[0].trunk[0].weight
        for m in methods[1:]:
            self.assertTrue(torch.equal(ref,m.trunk[0].weight))
            self.assertTrue(torch.equal(methods[0].adapter[0].weight,m.adapter[0].weight))

if __name__=="__main__":unittest.main(verbosity=2)
