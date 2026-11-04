import importlib.util,json,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("ma665",ROOT/"source"/"run_experiment.py")
RUNNER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUNNER)

class MA665EvidenceTests(unittest.TestCase):
    def test_grid_bytes_and_native_pca_alias(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        self.assertEqual(len(rows),144)
        self.assertEqual({r["seed"] for r in rows},{66501,66502})
        for r in rows:
            stem=f"{r['seed']}_n{r['filter_count']}_rho{r['residual_rms']:g}_k{r['rank']}_"
            p=ROOT/"source"/f"{stem}{r['method']}.npz"
            self.assertTrue(p.exists(),p);self.assertEqual(p.stat().st_size,r["serialized_bytes"])
            if r["method"]=="mirror_pca_code":
                a=np.load(p);b=np.load(ROOT/"source"/f"{stem}native_pca_code.npz")
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])

    def test_full_metric_replay(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        expected={(r["seed"],r["filter_count"],r["residual_rms"],r["rank"],r["method"]):r for r in rows}
        for seed in (66501,66502):
            for n in (16,64):
                for rho in (0.,.05,.2):
                    for rank in (4,8,16):
                        for r in RUNNER.run_condition(seed,n,rho,rank):
                            old=expected[(seed,n,rho,rank,r["method"])]
                            self.assertEqual(r["serialized_bytes"],old["serialized_bytes"])
                            self.assertAlmostEqual(r["output_nrmse"],old["output_nrmse"],places=12)
        self.assertFalse(json.loads((ROOT/"freeze.json").read_text())["fresh_accessed"])

if __name__=="__main__":unittest.main()
