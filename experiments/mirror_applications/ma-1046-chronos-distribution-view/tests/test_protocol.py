import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source" / "run_chronos.py"
spec = importlib.util.spec_from_file_location("run_chronos", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class MA1046ProtocolTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(15)
        self.samples=rng.normal(size=(5,32,24)).astype(np.float32)
        self.samples *= np.linspace(0.5,2.0,24,dtype=np.float32)[None,None,:]
        self.samples += np.linspace(-1.0,1.0,24,dtype=np.float32)[None,None,:]

    def test_givens_equals_normalized_hermite_residual(self):
        for theta in (-0.60,-0.25,0.0,0.35,0.60):
            actual,_=mod.mirror_view(self.samples,theta)
            alias=mod.hermite_control(self.samples,theta)
            np.testing.assert_allclose(actual,alias,rtol=2e-6,atol=2e-6)

    def test_view_preserves_per_lead_ensemble_mean_and_variance(self):
        for theta in (-0.5,0.0,0.5):
            out,_=mod.mirror_view(self.samples,theta)
            np.testing.assert_allclose(out.mean(axis=1),self.samples.mean(axis=1),rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(out.var(axis=1),self.samples.var(axis=1),rtol=2e-5,atol=2e-5)

    def test_frequency_interpolation_and_clamping(self):
        self.assertAlmostEqual(mod.theta_from_code(15,-0.4,0.2),-0.4)
        self.assertAlmostEqual(mod.theta_from_code(60,-0.4,0.2),0.2)
        self.assertAlmostEqual(mod.theta_from_code(30,-0.4,0.2),-0.1)
        self.assertAlmostEqual(mod.theta_from_code(1440,-0.4,0.2),0.2)

    def test_development_windows_are_chronological_and_disjoint(self):
        x=np.sin(np.arange(3000,dtype=np.float32))
        wins=mod.make_windows("ETTh1",[("series",x)],"dev")
        fit=[w for w in wins if w["split"]=="fit"]
        val=[w for w in wins if w["split"]=="val"]
        split=int(.60*len(x)); val_end=int(.80*len(x))
        self.assertEqual(len(fit),4)
        self.assertEqual(len(val),2)
        self.assertTrue(all(w["origin"]+w["horizon"]<=split for w in fit))
        self.assertTrue(all(w["origin"]>=split and w["origin"]+w["horizon"]<=val_end for w in val))
        spans=sorted((w["origin"],w["origin"]+w["horizon"]) for w in fit+val)
        for (_,end),(start,_) in zip(spans,spans[1:]):
            self.assertLessEqual(end,start)

    def test_crps_zero_for_perfect_degenerate_forecast(self):
        y=np.asarray([[1.,2.,3.]])
        s=np.repeat(y[:,None,:],4,axis=1)
        np.testing.assert_allclose(mod.sample_crps(s,y),0.0,atol=0.0)


if __name__=="__main__":
    unittest.main()
