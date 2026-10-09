import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from role_ops import givens_view,sparse_decode,fit_centroids,assign_roles

class RoleOpsTest(unittest.TestCase):
    def test_givens_is_orthogonal(self):
        torch.manual_seed(2); x=torch.randn(9,32); a=torch.randn(16)
        y=givens_view(x,a)
        self.assertTrue(torch.allclose(x.norm(dim=-1),y.norm(dim=-1),atol=2e-6))
        self.assertTrue(torch.allclose(givens_view(y,-a),x,atol=2e-6))
    def test_sparse_decode_backpropagates_to_view_angles(self):
        torch.manual_seed(7); code=torch.rand(6,32); angles=torch.nn.Parameter(torch.zeros(16)); dec=torch.randn(32,6); x=torch.randn(6,6);skip=torch.randn(6,6);bias=torch.randn(6)
        viewed=givens_view(code,angles); y,_,_=sparse_decode(viewed,dec,bias,skip,x,8); y.square().mean().backward()
        self.assertIsNotNone(angles.grad); self.assertGreater(float(angles.grad.abs().sum()),0.0)

    def test_sparse_decode_uses_exactly_k_atoms(self):
        torch.manual_seed(3); x=torch.randn(5,6); code=torch.rand(5,32); dec=torch.randn(32,6); b=torch.randn(6); skip=torch.randn(6,6)
        y,ids,vals=sparse_decode(code,dec,b,skip,x,8)
        self.assertEqual(ids.shape,(5,8)); self.assertEqual(vals.shape,(5,8)); self.assertEqual(y.shape,x.shape)
    def test_centroid_roles_deterministic(self):
        torch.manual_seed(4); x=torch.cat([torch.randn(20,4)-2,torch.randn(20,4)+2])
        c,l=fit_centroids(x,2,53401); self.assertEqual(len(torch.unique(l)),2)
        self.assertTrue(torch.equal(l,assign_roles(x,c)))

if __name__=='__main__': unittest.main()
