import importlib.util,json,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("ma670",ROOT/"source"/"run_experiment.py")
RUNNER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUNNER)

class MA670EvidenceTests(unittest.TestCase):
    def test_rows_bytes_and_vera_alias(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        self.assertEqual(len(rows),48)
        self.assertEqual({r['seed'] for r in rows},{67001,67002})
        for r in rows:
            stem=f"{r['seed']}_n{r['source_count']}_rho{r['residual_rms']:g}_"
            p=ROOT/"source"/f"{stem}{r['method']}.npz"
            self.assertTrue(p.exists(),p);self.assertEqual(p.stat().st_size,r['serialized_bytes'])
        for seed in (67001,67002):
            stem=f"{seed}_n64_rho0_"
            a=np.load(ROOT/"source"/f"{stem}mirror_shared_bank.npz");b=np.load(ROOT/"source"/f"{stem}native_vera_shared_bank.npz")
            for x,y in [('base','base'),('shared_left_atoms','shared_left'),('shared_right_atoms','shared_right'),('source_mirror_codes','source_scales'),('target_mirror_codes','composed_scales')]:np.testing.assert_array_equal(a[x],b[y])

    def test_metric_replay_and_fresh_lock(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        expected={(r['seed'],r['source_count'],r['residual_rms'],r['method']):r for r in rows}
        for seed in (67001,67002):
            for n in (16,64):
                for rho in (0.,.05,.2):
                    for r in RUNNER.run_condition(seed,n,rho):
                        old=expected[(seed,n,rho,r['method'])]
                        self.assertEqual(r['serialized_bytes'],old['serialized_bytes'])
                        self.assertAlmostEqual(r['target_output_nrmse'],old['target_output_nrmse'],places=12)
        self.assertFalse(json.loads((ROOT/"freeze.json").read_text())['fresh_accessed'])

if __name__=='__main__':unittest.main()
