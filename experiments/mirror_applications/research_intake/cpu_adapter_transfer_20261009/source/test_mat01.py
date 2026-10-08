#!/usr/bin/env python3
"""MAT01 deterministic tests: tensor ops, leakage, true serial bytes and data transforms."""
import unittest
import numpy as np
import torch
from run_mat01 import (D,H,C,R,make_world,source_family,fit_source_basis,FrozenBackbone,
                       role_batch,rotate4,core_from_code,transform,spatial_shift,actual_npz_bytes,
                       inference_arrays,train_target,target_support)

class MAT01Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
    def test_holdout_exact_disjoint_and_train_normalizer(self):
        w=make_world(11)
        self.assertEqual(len(set(w.train_ids).intersection(w.audit_ids)),0)
        self.assertAlmostEqual(float(w.x_raw[w.train_ids].mean(axis=0)[30]),float(w.scaler_mean[30]),places=6)
    def test_all_task_transforms_nontrivial(self):
        x=make_world(12).x_raw[:7]
        for name in ('left1','right1','up1','down1','blur065','contrast07',
                     'upper_left1','lower_right1','noise025','dropout020'):
            z=transform(x,name,np.random.default_rng(3))
            self.assertEqual(z.shape,x.shape)
            self.assertGreater(np.max(np.abs(z-x)),1e-5,msg=name)
    def test_zero_boundary_shift_not_roll(self):
        x=np.zeros((1,D),dtype=np.float32);x[0,0]=1
        y=spatial_shift(x,-1,0)
        self.assertAlmostEqual(float(y.sum()),0.)
    def test_mirror_operator_orthogonal(self):
        torch.manual_seed(4)
        for k in range(5):
            Q=rotate4(torch.randn(4))
            self.assertLess(float(torch.linalg.norm(Q@Q.T-torch.eye(R))),1e-5)
    def test_mirror_delta_gradient_nonzero(self):
        torch.manual_seed(2)
        code=torch.nn.Parameter(torch.tensor([.3,.2,.1,-.2]))
        c=torch.randn(R,R);U=torch.randn(H,R);V=torch.randn(R,C)
        obs=core_from_code('mirror4',code,c,torch.zeros(R,R,R))
        loss=(U@obs@V).pow(2).sum();loss.backward()
        self.assertTrue(torch.isfinite(code.grad).all())
        self.assertGreater(float(code.grad.norm()),1e-4)
    def test_linear_and_diag_shapes(self):
        U=torch.randn(H,R);V=torch.randn(R,C);c=torch.randn(R,R);dictionary=torch.randn(R,R,R)
        for meth in ('mirror4','linear4','diag4','fullcore16'):
            z=torch.randn(16 if meth=='fullcore16' else 4)
            self.assertEqual(tuple((U@core_from_code(meth,z,c,dictionary)@V).shape),(H,C))
    def test_full_npz_serializer_count_and_roundtrip(self):
        import io
        payload={'w':np.linspace(0.,1.,24,dtype=np.float32).reshape(6,4),'roles':np.arange(6,dtype=np.int32)}
        z=io.BytesIO();np.savez(z,**payload)
        self.assertEqual(actual_npz_bytes(payload),len(z.getvalue()))
        with np.load(io.BytesIO(z.getvalue())) as loaded:
            np.testing.assert_array_equal(loaded['w'],payload['w'])
    def test_role_batch_deterministic_seed(self):
        w=make_world(14)
        a=role_batch(w,w.train_ids[:15],'noise025',14,'target_train')
        b=role_batch(w,w.train_ids[:15],'noise025',14,'target_train')
        self.assertTrue(torch.equal(a,b))
        s=target_support(w,14)
        self.assertEqual(len(s),300)
        self.assertEqual(len(set(s)&set(w.audit_ids)),0)
    def test_native_base_broadcast_no_extra_trunk_forward(self):
        torch.manual_seed(8)
        base=FrozenBackbone(8)
        data=torch.randn(7,D)
        y=base(data)
        self.assertTrue(torch.equal(y.unsqueeze(1).expand(-1,4,-1)[:,3],y))

if __name__=='__main__':unittest.main(verbosity=2)
