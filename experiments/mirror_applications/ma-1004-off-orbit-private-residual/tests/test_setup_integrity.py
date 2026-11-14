import json, pathlib, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class MA1004SetupTests(unittest.TestCase):
    def test_fixed_disjoint_old_index_windows(self):
        p=json.loads((ROOT/'PROTOCOL.json').read_text())
        self.assertEqual(p['split']['development'],'uracil 100000 <= old_indices < 200000; select up to 128 deterministic rows for rank/shot/private-state selection')
        self.assertEqual(p['split']['fresh'],'uracil 500000 <= old_indices < 650000; select up to 64 deterministic rows after protocol and model/config/payload lock')
        self.assertFalse(p['split']['fresh_raw_or_labels_opened'])
        self.assertTrue(p['split']['fresh_file_downloaded_but_not_parsed_or_scored'])
        self.assertLess(100000,200000);self.assertLess(200000,500000)
    def test_provenance_contains_cc0_and_pinned_artifact_hashes(self):
        p=json.loads((ROOT/'runs/DATA_PROVENANCE.json').read_text())
        self.assertEqual(p['license'],'CC0-1.0')
        self.assertEqual({x['molecule'] for x in p['artifacts']},{'benzene','toluene','naphthalene','uracil'})
        self.assertTrue(all(len(x['sha256'])==64 for x in p['artifacts']))
        self.assertTrue((ROOT/'source/PA300_LICENSE').exists())
if __name__=='__main__':unittest.main()
