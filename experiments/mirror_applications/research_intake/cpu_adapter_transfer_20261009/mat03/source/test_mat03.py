#!/usr/bin/env python3
"""Local shape, label firewall, input identity and one-trunk-forward tests."""
import io
import unittest
import numpy as np
import torch
from run_mat03 import world,hadamard,patterns,MultiHeadMirror,METHODS,K,F,TASKS,labels_from_images,metrics

class MAT03Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_hadamard_orthogonal(self):
        H=hadamard(32)
        self.assertLess(float(torch.linalg.norm(H@H.T-torch.eye(32))),1e-6)
    def test_splits_seed_deterministic(self):
        a=world(31);b=world(31)
        for i in range(4):self.assertTrue(torch.equal(a[i],b[i]))
        self.assertTrue(len(a[0])>len(a[2]))
        self.assertEqual(tuple(a[1].shape[1:]),(K,))
    def test_distinct_label_tasks_and_balance(self):
        x,y,_,_,_,_,_,_=world(32)
        self.assertEqual(len(TASKS),5)
        for j in range(K):
            p=float(y[:,j].float().mean())
            self.assertGreater(p,.25);self.assertLess(p,.75)
        self.assertGreater(sum(bool(not torch.equal(y[:,0],y[:,j])) for j in range(1,K)),2)
    def test_exact_same_image_input_for_every_role(self):
        trainx,trainy,testx,testy,*_=world(33)
        model=MultiHeadMirror('native_multihead',33)
        self.assertEqual(model(testx[:13]).shape,(13,K,2))
        self.assertEqual(testx.ndim,2)
        # The model.forward API accepts a single x, never a per-role x list.
    def test_all_method_shapes_and_one_trunk_forward(self):
        for method in METHODS:
            model=MultiHeadMirror(method,31)
            x=torch.randn(5,64)
            observed={'n':0}
            def hook(_a,_b,_c):observed['n']+=1
            z=model.trunk.register_forward_hook(hook)
            with torch.no_grad():out=model(x)
            z.remove()
            self.assertEqual(tuple(out.shape),(5,K,2),method)
            self.assertEqual(observed['n'],1,method)
            self.assertTrue(torch.isfinite(out).all())
    def test_shared_method_identical_logical_outputs(self):
        model=MultiHeadMirror('shared',31)
        y=model(torch.randn(3,64))
        self.assertTrue(torch.equal(y[:,0],y[:,4]))
    def test_zero_output_views_preserve_function(self):
        x=torch.randn(11,64)
        for method in ['mirror_hadamard4','native_hadamard_linear4','native_direct_diagonal4']:
            model=MultiHeadMirror(method,31)
            with torch.no_grad():model.codes.zero_();y=model(x);ref=model.common(model.trunk(x))
            self.assertLess(float((y-ref.unsqueeze(1)).abs().max()),2e-6,method)
    def test_role_code_gradient_nonzero(self):
        model=MultiHeadMirror('mirror_hadamard4',31)
        x=torch.randn(13,64)
        task=torch.randint(0,2,(13,K))
        z=model(x)
        loss=torch.nn.functional.cross_entropy(z.reshape(-1,2),task.reshape(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(model.codes.grad).all())
        self.assertGreater(float(model.codes.grad.abs().max()),1e-7)
    def test_actual_npz_bytes_and_fixed_basis_provenance(self):
        tr,yt,te,ye,sha,thresholds,mean,sd=world(33)
        size=[]
        for meth in ('native_multihead','mirror_hadamard4','native_hadamard_linear4'):
            m=MultiHeadMirror(meth,33)
            size.append(m.serialized_bytes(mean,sd,thresholds))
        self.assertTrue(all(s>1000 for s in size))
        self.assertLess(size[1],size[0])
    def test_pattern_seed_repeat(self):
        a,b=patterns();x,y=patterns()
        self.assertTrue(torch.equal(a,x));self.assertTrue(torch.equal(b,y))

if __name__=='__main__':unittest.main(verbosity=2)
