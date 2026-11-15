import hashlib,json,pathlib,unittest
import numpy as np
E=pathlib.Path(__file__).resolve().parents[1];D=json.loads((E/'runs'/'development.json').read_text()) if (E/'runs'/'development.json').exists() else None
class MA1026Tests(unittest.TestCase):
    def test_two_modality_codes_are_present(self):
        if D is None:self.skipTest('result payloads not generated yet')
        raw=(E/'runs'/'payload_mirror_maple.bin').read_bytes();meta=json.loads(raw.split(b'\n',1)[0])
        self.assertIn('text and vision Givens angles',meta['prompt_mapping'])
        self.assertEqual(meta['prompt_depth'],2);self.assertEqual(meta['n_context'],2)
    def test_payload_hashes_and_full_bundle_arithmetic(self):
        if D is None:self.skipTest('result payloads not generated yet')
        for name,n in D['prompt_payload_bytes'].items():
            raw=(E/'runs'/f'payload_{name}.bin').read_bytes();self.assertEqual(len(raw),n)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),D['payload_sha256'][name])
            self.assertEqual(D['full_bundle_bytes'][name],D['backbone_bytes']+n)
    def test_audit_index_disjointness_and_hash_count(self):
        if D is None:self.skipTest('result payloads not generated yet')
        self.assertTrue(D['fresh_opened_after_all_prompt_states_frozen'])
        self.assertEqual(sum(map(len,D['audit_image_sha256'].values())),40)
        for cls,ids in D['audit_indices'].items():
            self.assertFalse(set(ids)&set(D['excluded_MA1024_MA1025_test_indices'][cls]))
    def test_additive_control_is_not_beaten(self):
        if D is None:self.skipTest('result payloads not generated yet')
        self.assertGreater(D['results']['additive_rank1_multimodal']['harmonic_mean'],D['results']['mirror_maple']['harmonic_mean'])
    def test_rotation_preserves_joint_norm(self):
        x=np.arange(2*768,dtype=np.float64).reshape(2,768)/100
        a=.314159;c,s=np.cos(a),np.sin(a);y=np.stack((c*x[0]-s*x[1],s*x[0]+c*x[1]))
        self.assertAlmostEqual(np.linalg.norm(x),np.linalg.norm(y),places=9)
if __name__=='__main__':unittest.main()
