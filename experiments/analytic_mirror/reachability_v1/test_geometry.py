"""Finite algebra fixtures; NOT Transformer training or capacity measurements."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path
import sys
import unittest
import numpy as np
import scipy
from scipy.linalg import expm, eigh
from scipy.special import ndtr
import local_geometry as geo

METRICS = {}

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261005)

    def test_mirror_tangent_and_generator_gradient(self):
        v, d = self.rng.normal(size=(2, 6))
        a = self.rng.normal(size=(6, 6)) / 4
        tangent = geo.mirror_tangent(v, a)
        self.assertIsNotNone(tangent)
        eps = 1e-5
        fd = (geo.mirror(v,a,eps)-geo.mirror(v,a,-eps))/(2*eps)
        error = float(np.max(np.abs(tangent-fd)))
        self.assertLess(error, 1e-8)
        sensitivity = geo.generator_sensitivity(v,d)
        self.assertIsNotNone(sensitivity)
        self.assertAlmostEqual(float(d@tangent),float(np.sum(sensitivity*a)),12)
        METRICS['tangent_central_difference_max_abs'] = error

    def test_fixed_view_folding(self):
        w1 = self.rng.normal(size=(6,4)); w2 = self.rng.normal(size=(4,6))
        b1 = self.rng.normal(size=6); b2 = self.rng.normal(size=4)
        x = self.rng.normal(size=4); a = self.rng.normal(size=(6,6))/4
        q = expm(a); qi = expm(-a)
        value = geo.mirror(w1@x+b1,a)
        self.assertIsNotNone(value)
        original = w2@value+b2
        v = (q@w1)@x+q@b1
        folded = (w2@qi)@(v*ndtr(v))+b2
        np.testing.assert_allclose(original,folded,rtol=1e-12,atol=1e-12)
        METRICS['fixed_view_fold_max_abs'] = float(np.max(np.abs(original-folded)))

    def test_parameter_tangent_emulates_fixed_mirror(self):
        w1 = self.rng.normal(size=(6,4)); w2 = self.rng.normal(size=(4,6))
        b1 = self.rng.normal(size=6); x = self.rng.normal(size=4)
        a = self.rng.normal(size=(6,6))/4
        tangent = geo.mirror_tangent(w1@x+b1,a)
        self.assertIsNotNone(tangent)
        def forward(e):
            v=(w1+e*a@w1)@x+(b1+e*a@b1)
            return (w2-e*w2@a)@(v*ndtr(v))
        eps=1e-5
        fd=(forward(eps)-forward(-eps))/(2*eps)
        err=float(np.max(np.abs(fd-w2@tangent)))
        self.assertLess(err,1e-7)
        METRICS['base_emulation_tangent_max_abs']=err

    def test_weighted_minimum_reachability(self):
        j=self.rng.normal(size=(3,5)); a=self.rng.normal(size=(5,5))
        metric=a.T@a+np.eye(5); target=self.rng.normal(size=3)
        out=geo.weighted_reachability(j,metric,target)
        self.assertIsNotNone(out)
        gram=j@np.linalg.solve(metric,j.T)
        exact=np.linalg.solve(metric,j.T)@np.linalg.solve(gram,target)
        np.testing.assert_allclose(out['step'],exact,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(j@out['step'],target,rtol=1e-10,atol=1e-10)
        self.assertAlmostEqual(out['cost'],float(target@np.linalg.solve(gram,target)/2),10)
        METRICS['weighted_minimum_constraint_residual']=float(np.linalg.norm(j@out['step']-target))

    def test_unreachable_is_not_zero_cost(self):
        j=np.diag([1.,0.]); target=np.array([0.,1.])
        out=geo.weighted_reachability(j,np.eye(2),target)
        self.assertIsNotNone(out)
        self.assertFalse(out['reachable']); self.assertTrue(np.isinf(out['cost']))
        self.assertAlmostEqual(out['residual_norm'],1.0)
        feasible=geo.weighted_reachability(j,np.eye(2),np.array([2.,0.]))
        self.assertTrue(feasible['reachable']); self.assertAlmostEqual(feasible['cost'],2.0)

    def test_expensive_direction_is_in_numerator(self):
        out=geo.efficiency_directions(np.diag([1.,0.1]),np.eye(2),np.eye(2),np.eye(2))
        self.assertIsNotNone(out)
        values,vectors,cost_matrix=out
        np.testing.assert_allclose(values,[100.,1.],atol=1e-10)
        self.assertAlmostEqual(abs(vectors[1,0]),1.0)
        wrong_values,wrong_vectors=eigh(np.eye(2),cost_matrix)
        self.assertAlmostEqual(abs(wrong_vectors[0,-1]),1.0)
        METRICS['efficiency_eigenvalues']=values.tolist()
        METRICS['equal_target_base_costs']=[0.5,50.0]

    def test_coordinate_change_preserves_cost(self):
        j=self.rng.normal(size=(3,5)); a=self.rng.normal(size=(5,5))
        metric=a.T@a+np.eye(5); target=self.rng.normal(size=3)
        transform=self.rng.normal(size=(5,5))+4*np.eye(5)
        old=geo.weighted_reachability(j,metric,target)
        self.assertIsNotNone(old)
        new=geo.weighted_reachability(j@transform,transform.T@metric@transform,target)
        self.assertAlmostEqual(old['cost'],new['cost'],10)
        METRICS['coordinate_cost_abs_difference']=abs(old['cost']-new['cost'])

    def test_old_task_penalty_is_accounted_on_both_sides(self):
        j=np.eye(2); old=np.array([[0.,1.]]); target=np.array([0.,1.])
        metric=np.eye(2)+9*old.T@old
        out=geo.weighted_reachability(j,metric,target)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out['cost'],5.0)
        values,_,_=geo.efficiency_directions(j,metric,np.eye(2),metric)
        np.testing.assert_allclose(values,np.ones(2),atol=1e-12)

    def test_frozen_preconditioned_gradient_dynamics(self):
        j=np.diag([1.,0.1]); metric=np.diag([2.,1.]); target=np.ones(2)
        lr=0.5; steps=20
        analytical=geo.frozen_gradient_residual(j,metric,target,lr,steps)
        self.assertIsNotNone(analytical)
        delta=np.zeros(2)
        for _ in range(steps):
            delta-=lr*np.linalg.solve(metric,j.T@(j@delta-target))
        np.testing.assert_allclose(analytical,target-j@delta,atol=1e-13)
        METRICS['twenty_step_residual']=analytical.tolist()

    def test_simplex_minimum_support_moments(self):
        for rank in [1,2,4,8]:
            c=geo.regular_simplex(rank)
            self.assertIsNotNone(c)
            self.assertEqual(c.shape,(rank+1,rank))
            np.testing.assert_allclose(c.mean(axis=0),0,atol=1e-14)
            np.testing.assert_allclose(c.T@c/(rank+1),np.eye(rank)/rank,atol=1e-14)
            expected=np.full((rank+1,rank+1),-1/rank); np.fill_diagonal(expected,1)
            np.testing.assert_allclose(c@c.T,expected,atol=1e-14)
        METRICS['simplex_ranks_checked']=[1,2,4,8]

    def test_token_cycle_does_not_cancel_first_order(self):
        coefficients=np.array([1.,-1.]); sensitivities=np.array([1.,2.])
        self.assertEqual(float(coefficients.sum()),0.0)
        self.assertEqual(float(sensitivities@coefficients),-1.0)
        offsets=np.stack([coefficients,-coefficients])
        self.assertEqual(float(np.mean(offsets@sensitivities)),0.0)
        METRICS['token_cycle_first_order_counterexample']=-1.0

    def test_cross_token_curvature_cannot_be_dropped(self):
        h=np.ones((2,2)); c=np.array([1.,-1.])
        exact=float(c@h@c); diagonal=float(c@np.diag(np.diag(h))@c)
        self.assertEqual(exact,0.0); self.assertEqual(diagonal,2.0)
        METRICS['cross_token_exact_vs_diagonal']=[exact,diagonal]

    def test_volume_preserving_does_not_bound_condition_number(self):
        q=expm(np.diag([3.,-3.]))
        self.assertAlmostEqual(np.linalg.det(q),1.0)
        self.assertAlmostEqual(np.linalg.cond(q),np.exp(6),10)
        METRICS['det_one_condition_number']=float(np.linalg.cond(q))

    def test_centered_quadratic_prediction_only_with_fixed_sensitivities(self):
        c=geo.regular_simplex(4)
        self.assertIsNotNone(c)
        h=np.diag([1.,2.,3.,4.]); linear=np.array([2.,-3.,4.,-5.]); rho=0.1
        measured=np.mean([linear@(rho*z)+0.5*(rho*z)@h@(rho*z) for z in c])
        predicted=0.5*rho*rho*np.trace(h)/4
        self.assertAlmostEqual(measured,predicted,14)
        METRICS['centered_quadratic_loss']=float(measured)

    def test_input_validation(self):
        # This checks actual public numerical interfaces, not unverified model training.
        self.assertIsNotNone(geo.regular_simplex(1))
        with self.assertRaises(ValueError): geo.regular_simplex(0)
        with self.assertRaises(ValueError): geo.weighted_reachability(np.eye(2),np.diag([1.,-1.]),np.ones(2))
        with self.assertRaises(ValueError): geo.efficiency_directions(np.diag([1.,0.]),np.eye(2),np.eye(2),np.eye(2))
        with self.assertRaises(ValueError): geo.frozen_gradient_residual(np.eye(2),np.eye(2),np.ones(2),3.0,2)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    here=Path(__file__).resolve().parent
    payload={'scope':'finite analytical regression fixtures; no Transformer replay or training',
             'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
             'platform':platform.platform(),'dtype':'float64',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'passed':result.wasSuccessful(),'metrics':METRICS,
             'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [here/'local_geometry.py',Path(__file__).resolve()]}}
    (here/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
