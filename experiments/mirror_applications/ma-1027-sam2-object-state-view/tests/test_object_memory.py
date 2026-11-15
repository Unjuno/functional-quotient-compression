import hashlib,json,pathlib,unittest
import numpy as np
E=pathlib.Path(__file__).resolve().parents[1];R=E/'runs';D=json.loads((R/'development.json').read_text()) if (R/'development.json').exists() else None
class SAM2MemoryTests(unittest.TestCase):
    def test_checkpoint_and_scene_provenance(self):
        if D is None:self.skipTest('run pending')
        self.assertEqual(D['checkpoint_sha256'],'7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69')
        self.assertEqual(D['scene']['seed'],102701);self.assertEqual(len(D['scene']['rows']),5)
    def test_actual_session_payloads_and_hashes(self):
        if D is None:self.skipTest('run pending')
        for method,row in D['results'].items():
            p=R/f'payload_{method}.bin';raw=p.read_bytes()
            self.assertEqual(len(raw),row['session_payload_bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['payload_sha256'])
            self.assertEqual(row['full_session_payload_bytes'],D['checkpoint_bytes']+D['config_bytes']+D['common_video_cache_bytes']+len(raw))
    def test_four_object_predictions_saved(self):
        if D is None:self.skipTest('run pending')
        for method in D['results']:
            a=np.load(R/f'predictions_{method}.npz')['masks']
            self.assertEqual(a.shape,(5,4,256,256))
    def test_rank1_control_beats_mirror_quality(self):
        if D is None:self.skipTest('run pending')
        self.assertGreater(D['results']['additive_rank1']['mean_jf'],D['results']['mirror_givens']['mean_jf']+.01)
    def test_mirror_misses_its_frozen_quality_gate(self):
        if D is None:self.skipTest('run pending')
        self.assertLess(D['results']['mirror_givens']['mean_jf'],D['results']['independent']['mean_jf']-.01)
        self.assertLess(D['derived_comparisons']['mirror_full_session_savings_vs_independent'],.25)
    def test_memory_reconstruction_diagnostic(self):
        if D is None:self.skipTest('run pending')
        m=D['memory_reconstruction']
        self.assertLess(abs(m['mirror_givens']['mean_relative_rmse']-m['hard_shared']['mean_relative_rmse']),.001)
if __name__=='__main__':unittest.main()
