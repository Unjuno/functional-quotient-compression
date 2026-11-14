import json, pathlib, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class MA1004SetupTests(unittest.TestCase):
    def test_fixed_disjoint_old_index_windows_and_access_record(self):
        p=json.loads((ROOT/'PROTOCOL.json').read_text())
        self.assertEqual(p['split']['development'],'uracil 80000 <= old_indices < 100000; select up to 128 deterministic rows for rank/shot/private-state selection')
        self.assertEqual(p['split']['fresh'],'uracil 100000 <= old_indices < 133770; select up to 64 deterministic rows after protocol and model/config/payload lock')
        self.assertTrue(p['split']['fresh_shard_deserialized_for_old_indices'])
        self.assertFalse(p['split']['fresh_label_values_inspected'])
        self.assertTrue(p['split']['fresh_tensor_file_deserialized'])
        self.assertFalse(p['split']['fresh_metrics_or_quality_opened'])
        self.assertFalse(p['split']['locked_before_any_fresh_shard_access'])
        self.assertTrue(p['split']['locked_before_metric_access'])
        row=json.loads((ROOT/'runs/ROW_MANIFEST.json').read_text())
        self.assertFalse(row['fresh_labels_opened'])
        self.assertTrue(row['splits'][-1]['selected_rows'])
        self.assertEqual(row['splits'][-1]['old_index_window'],[100000,133770])
    def test_provenance_contains_cc0_and_pinned_artifact_hashes(self):
        p=json.loads((ROOT/'runs/DATA_PROVENANCE.json').read_text())
        self.assertEqual(p['license'],'CC0-1.0')
        self.assertEqual({x['molecule'] for x in p['artifacts']},{'benzene','toluene','naphthalene','uracil'})
        self.assertTrue(all(len(x['sha256'])==64 for x in p['artifacts']))
        self.assertTrue((ROOT/'source/PA300_LICENSE').exists())
if __name__=='__main__':unittest.main()
