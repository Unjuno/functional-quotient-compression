import importlib.util,json,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("ma669",ROOT/"source"/"run_experiment.py")
RUNNER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUNNER)

class MA669EvidenceTests(unittest.TestCase):
    def test_grid_payload_sizes_and_native_alias(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        self.assertEqual(len(rows),216)
        self.assertEqual({r["seed"] for r in rows},{66901,66902})
        for r in rows:
            stem=f"{r['seed']}_n{r['task_count']}_shots{r['support_examples']}_rho{r['residual_rms']:g}_"
            p=ROOT/"source"/f"{stem}{r['method']}.npz"
            self.assertTrue(p.exists(),p);self.assertEqual(p.stat().st_size,r["serialized_bytes"])
        for seed in (66901,66902):
            r=next(x for x in rows if x['seed']==seed and x['task_count']==64 and x['support_examples']==4 and x['residual_rms']==0 and x['method']=='mirror_code')
            stem=f"{seed}_n64_shots4_rho0_"
            a=np.load(ROOT/"source"/f"{stem}mirror_code.npz");b=np.load(ROOT/"source"/f"{stem}native_lowrank_code.npz")
            for k in ('shared_base','atom_left','atom_right','coefficient_basis'):
                np.testing.assert_array_equal(a[k],b[k])
            np.testing.assert_array_equal(a['mirror_codes'],b['task_codes'])

    def test_selected_metric_replay_and_amendment(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        expected={(r['seed'],r['task_count'],r['support_examples'],r['residual_rms'],r['method']):r for r in rows}
        for seed in (66901,66902):
            for n in (16,64):
                for rho in (0.,.1,.25):
                    for r in RUNNER.run_condition(seed,n,4,rho):
                        old=expected[(seed,n,4,rho,r['method'])]
                        self.assertEqual(r['serialized_bytes'],old['serialized_bytes'])
                        self.assertAlmostEqual(r['heldout_nrmse'],old['heldout_nrmse'],places=12)
        freeze=json.loads((ROOT/'freeze.json').read_text())
        self.assertFalse(freeze['fresh_accessed']);self.assertEqual(len(freeze['amendments']),2)
        for r in rows:
            if r['method']=='lorahub_search':self.assertGreater(r['function_evaluations'],0)
            else:self.assertEqual(r['function_evaluations'],0)
        attempt=json.loads((ROOT/'source'/'attempts'/'initial_coordinate_search'/'development.json').read_text())
        self.assertEqual(len(attempt),216)

if __name__=='__main__':unittest.main()
