import importlib.util
import math
import os
import sys
import tempfile
import zipfile
import unittest
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "source" / "run_chronos.py"
spec = importlib.util.spec_from_file_location("run_chronos", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.path.insert(0,str(SRC.parent))
import evaluate_audit as audit_eval


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

    def test_wql_returns_pooled_loss_numerator_and_denominator(self):
        y=np.asarray([[1.0,2.0],[0.01,0.02]],dtype=np.float32)
        samples=np.repeat(y[:,None,:],4,axis=1)
        num,den,mae,cov=mod.quantile_metrics(samples,y)
        self.assertEqual(num.shape,(2,))
        np.testing.assert_allclose(den,3*np.sum(np.abs(y),axis=1))
        np.testing.assert_allclose(num,0.0,atol=0.0)

    def test_tsf_reader_skips_preamble_until_data_marker(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/"tiny.zip"
            content=("# Dataset Information\n@relation x\n@attribute series_name string\n"
                     "@frequency daily\n@missing false\n@equallength true\n@data\n"
                     "T1:1,2,3\nT2:4,5,6\n").encode()
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("tiny.tsf",content)
            count,rows=mod.tsf_selected(archive,"tiny.tsf",[1])
            self.assertEqual(count,2)
            self.assertEqual(rows[0][0],"1:T2")
            np.testing.assert_array_equal(rows[0][1],np.asarray([4,5,6],dtype=np.float32))

    def test_audit_series_aggregation_selects_window_rows(self):
        records=[{"dataset":"D","series":"A"},{"dataset":"D","series":"A"},{"dataset":"D","series":"B"}]
        rows=[{"crps":1.,"wql_num":2.,"wql_den":4.,"mase":.5,"coverage90":.8,"naive_crps":2.},
              {"crps":3.,"wql_num":3.,"wql_den":6.,"mase":1.5,"coverage90":.9,"naive_crps":3.},
              {"crps":4.,"wql_num":8.,"wql_den":8.,"mase":2.,"coverage90":.7,"naive_crps":4.}]
        out=audit_eval.series_normalized(records,rows)["D"]
        self.assertEqual(out["A"]["crps"],2.0)
        self.assertAlmostEqual(out["A"]["wql"],5/10)
        self.assertEqual(out["B"]["crps"],4.0)

    def test_crps_zero_for_perfect_degenerate_forecast(self):
        y=np.asarray([[1.,2.,3.]])
        s=np.repeat(y[:,None,:],4,axis=1)
        np.testing.assert_allclose(mod.sample_crps(s,y),0.0,atol=0.0)


if __name__=="__main__":
    unittest.main()
