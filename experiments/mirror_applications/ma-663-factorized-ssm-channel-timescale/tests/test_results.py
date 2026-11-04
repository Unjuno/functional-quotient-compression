import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("ma663",ROOT/"source"/"run_experiment.py")
RUNNER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUNNER)

class MA663EvidenceTests(unittest.TestCase):
    def test_all_payload_sizes_and_control_aliases(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        self.assertEqual(len(rows),240)
        self.assertEqual({r["seed"] for r in rows},{66301,66302})
        for row in rows:
            stem=f"{row['seed']}_n{row['task_count']}_rho{row['residual_rms']:g}_r{row['private_rank']}_"
            path=ROOT/"source"/f"{stem}{row['method']}.npz"
            self.assertTrue(path.exists(),path)
            self.assertEqual(path.stat().st_size,row["serialized_bytes"])
            if row["method"] in ("mirror_cp","mirror_cp_private"):
                native="native_cp" if row["method"]=="mirror_cp" else "native_cp_private"
                a=np.load(ROOT/"source"/f"{stem}{row['method']}.npz")
                b=np.load(ROOT/"source"/f"{stem}{native}.npz")
                for key in a.files: np.testing.assert_array_equal(a[key],b[key])

    def test_metric_replay_and_fresh_lock(self):
        rows=json.loads((ROOT/"source"/"development.json").read_text())
        expected={(r["seed"],r["task_count"],r["residual_rms"],r["private_rank"],r["method"]):r for r in rows}
        for seed in (66301,66302):
            for n in (16,64):
                for rho in (0.,.05,.2):
                    for rank in (0,2,4,8):
                        for row in RUNNER.run_condition(seed,n,rho,rank):
                            old=expected[(seed,n,rho,rank,row["method"])]
                            self.assertEqual(row["serialized_bytes"],old["serialized_bytes"])
                            self.assertAlmostEqual(row["output_nrmse"],old["output_nrmse"],places=12)
        freeze=json.loads((ROOT/"freeze.json").read_text())
        self.assertFalse(freeze["fresh_accessed"])
        self.assertTrue(freeze["amendments"])

if __name__=="__main__": unittest.main()
