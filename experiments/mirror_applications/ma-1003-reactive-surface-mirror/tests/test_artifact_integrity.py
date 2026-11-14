import hashlib, json, pathlib, unittest
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]

class MA1003ArtifactTests(unittest.TestCase):
    def test_locked_manifest_hash_and_fresh_is_unopened(self):
        protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
        manifest_path=ROOT/protocol['split_manifest']['path']
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(), protocol['split_manifest']['sha256'])
        self.assertFalse(protocol['fresh']['raw_files_opened'])
        self.assertEqual(protocol['fresh']['stage'],'6_new_datapoints_4')
        manifest=json.loads(manifest_path.read_text())
        dev_ids=set()
        for role,stages in manifest['roles'].items():
            for stage,facets in stages.items():
                if role=='fresh_audit':
                    continue
                for groups in facets.values():
                    for group in groups:
                        dev_ids.update(e['entry_id'] for e in group['entries'])
        fresh_ids={e['entry_id'] for facets in manifest['roles']['fresh_audit'].values() for groups in facets.values() for group in groups for e in group['entries']}
        self.assertFalse(dev_ids & fresh_ids)

    def test_mirror_and_native_low_rank_payload_are_exact_alias(self):
        mirror=(ROOT/'runs/mirror_payload.npz').read_bytes()
        native=(ROOT/'runs/native_factorized_payload.npz').read_bytes()
        self.assertEqual(mirror,native)
        self.assertEqual(len(mirror),5187)
        with np.load(ROOT/'runs/mirror_payload.npz',allow_pickle=False) as m, np.load(ROOT/'runs/native_factorized_payload.npz',allow_pickle=False) as n:
            self.assertEqual(set(m.files),set(n.files))
            for key in m.files:
                np.testing.assert_array_equal(m[key],n[key])
            self.assertEqual(m['B'].shape,(257,2))
            self.assertEqual(m['codes'].shape,(4,2))

    def test_development_gate_is_missed_on_force_and_independent_quality(self):
        data=json.loads((ROOT/'runs/DEV_SCREEN.json').read_text())
        self.assertEqual(data['selected'],{'rank':2,'shots':4,'dev_energy_mae_eV_atom':0.013056032951811045})
        for row in data['dev_selected_metrics']:
            self.assertGreater(row['mirror_force_component_mae_eV_A'],row['frozen_force_component_mae_eV_A'])
            self.assertGreater(row['mirror_force_component_mae_eV_A'],row['independent_force_component_mae_eV_A'])
            self.assertGreater(row['mirror_energy_mae_eV_atom'],row['independent_energy_mae_eV_atom'])
        self.assertLess(data['payloads']['byte_ratio_mirror_to_independent'],0.75)

if __name__=='__main__': unittest.main()
