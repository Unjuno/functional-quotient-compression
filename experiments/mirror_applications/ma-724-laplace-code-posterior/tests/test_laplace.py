import importlib.util
import unittest
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma724',ROOT/'source'/'run_laplace.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class LaplaceTests(unittest.TestCase):
    def test_givens_is_norm_preserving(self):
        model=m.MirrorMLP();h=torch.randn(9,m.D);z=model.view(h)
        self.assertTrue(torch.allclose(torch.linalg.norm(h,dim=-1),torch.linalg.norm(z,dim=-1),atol=1e-5))
    def test_split_replay_and_standardizer_train_only(self):
        (a,b,c,d,e,f),scaler,digest=m.dataset()
        self.assertEqual((len(a),len(c),len(e)),(341,114,114))
        self.assertEqual(a.shape[1],30);self.assertEqual(len(digest),64)
        self.assertTrue(torch.isfinite(a).all())
        A,B,C=(set(map(tuple,z.numpy())) for z in (a,c,e))
        self.assertEqual(len(A&B),0);self.assertEqual(len(A&C),0);self.assertEqual(len(B&C),0)
    def test_metrics_are_finite(self):
        x=m.metrics([.2,.7,.6,.1],[0,1,1,0])
        self.assertTrue(all(0<=x[k]<=1 for k in ['brier','ece10','accuracy']))
        self.assertGreaterEqual(x['nll'],0)
    def test_development_payloads_reconstruct_shared_model(self):
        base=torch.load(ROOT/'source'/'shared_base.pt',map_location='cpu',weights_only=False)
        code=torch.load(ROOT/'source'/'posterior_mirror_full.pt',map_location='cpu',weights_only=False)
        model=m.MirrorMLP();state=dict(base['state_dict']);state['m']=code['m_mean'];model.load_state_dict(state)
        self.assertEqual(tuple(code['posterior_covariance'].shape),(m.D//2,m.D//2))
        self.assertTrue(torch.isfinite(model(torch.zeros(3,30))).all())
        summary=json.loads((ROOT/'source'/'development_result.json').read_text())
        self.assertEqual((ROOT/'source'/'shared_base.pt').stat().st_size,summary['shared_base_bytes'])

    def test_curvature_is_symmetric_positive_semidefinite(self):
        torch.manual_seed(4);model=m.MirrorMLP();X=torch.randn(12,30)
        diag,h=m.fisher_and_subnet(model,X)
        for key in ['mirror_full','subnet_full']:
            H=h[key];self.assertTrue(torch.allclose(H,H.T,atol=1e-5));self.assertGreaterEqual(float(torch.linalg.eigvalsh(H).min()),-1e-5)
        self.assertTrue(all(torch.isfinite(v).all() for v in diag.values()))
if __name__=='__main__':unittest.main()
