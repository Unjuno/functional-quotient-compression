import hashlib, json, pathlib, unittest
import numpy as np

E=pathlib.Path(__file__).resolve().parents[1]
D=json.loads((E/'runs'/'development.json').read_text())

class EvidenceTests(unittest.TestCase):
    def test_payload_sizes_and_hashes(self):
        for name,size in D['prompt_payload_bytes'].items():
            p=E/'runs'/f'payload_{name}.bin'
            self.assertEqual(p.stat().st_size,size)
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),D['payload_sha256'][name])

    def test_payloads_include_inference_metadata(self):
        for name in D['prompt_payload_bytes']:
            raw=(E/'runs'/f'payload_{name}.bin').read_bytes()
            meta=json.loads(raw.split(b'\n',1)[0])
            self.assertEqual(meta['method'],name)
            self.assertIn('token_template',meta)
            if name=='mirror':
                self.assertEqual(meta['conditioner_and_view']['token_pairs'],[[0,1],[2,3]])

    def test_backbone_bundle_arithmetic(self):
        self.assertEqual(sum(x['bytes'] for x in D['model_files']),D['backbone_bytes'])
        for method,size in D['prompt_payload_bytes'].items():
            self.assertEqual(D['full_bundle_bytes'][method],D['backbone_bytes']+size)

    def test_audit_ids_are_disjoint_from_ma1024(self):
        for cls,ids in D['audit_indices'].items():
            self.assertFalse(set(ids)&set(D['excluded_MA1024_audit_indices'][cls]))
        for cls,ids in D['train_indices'].items():
            self.assertFalse(set(ids)&set(D['excluded_MA1024_train_indices'][cls]))

    def test_specificity_gate_fails_on_additive_tie(self):
        m=D['results']['mirror']; a=D['results']['additive_rank2']
        for metric in ('base_accuracy','novel_accuracy','harmonic_mean'):
            self.assertEqual(m[metric],a[metric])
        self.assertGreater(D['prompt_payload_bytes']['mirror'],0)

    def test_rotation_is_norm_preserving(self):
        # Two independent planar Givens rotations preserve the joint context norm.
        x=np.arange(4*512,dtype=np.float64).reshape(4,512)/1000
        p=np.array([.31,-.72]); y=x.copy()
        for k,(i,j) in enumerate(((0,1),(2,3))):
            c,s=np.cos(p[k]),np.sin(p[k]); y[i]=c*x[i]-s*x[j]; y[j]=s*x[i]+c*x[j]
        self.assertAlmostEqual(float(np.linalg.norm(x)),float(np.linalg.norm(y)),places=10)

if __name__=='__main__': unittest.main()
