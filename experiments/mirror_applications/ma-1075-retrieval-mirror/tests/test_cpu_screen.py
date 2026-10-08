import importlib.util
import json
import unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ma1075', ROOT/'source'/'run_cpu_screen.py')
ma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ma)

class CpuScreenTests(unittest.TestCase):
    def test_reference_jagged_ops_roundtrip_and_gradient(self):
        ma.register_reference_jagged_ops()
        x = torch.arange(12,dtype=torch.float32).reshape(6,2).requires_grad_()
        offsets=torch.tensor([0,2,6])
        dense=torch.ops.fbgemm.jagged_to_padded_dense(x,[offsets],[4],padding_value=0.0)
        recovered=torch.ops.fbgemm.dense_to_jagged(dense,[offsets])[0]
        self.assertTrue(torch.equal(recovered,x))
        recovered.sum().backward()
        self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))
        self.assertTrue(torch.equal(torch.ops.fbgemm.asynchronous_complete_cumsum(torch.tensor([2,4])),offsets))

    def test_givens_preserves_query_norm_and_matches_code_budget(self):
        givens=ma.RoleHead(18,'givens'); gain=ma.RoleHead(18,'pair_gain')
        self.assertEqual(sum(p.numel() for p in givens.parameters()),sum(p.numel() for p in gain.parameters()))
        x=torch.randn(6,ma.DIM); roles=torch.tensor([0,1,2,3,4,5])
        y=givens.transform(x,roles)
        self.assertTrue(torch.allclose(torch.linalg.norm(x,dim=-1),torch.linalg.norm(y,dim=-1),atol=1e-5))

    def test_development_metric_replays_from_saved_per_role_values(self):
        result=json.loads((ROOT/'source'/'development_seed41075.json').read_text())
        for run in result['runs']:
            macro=sum(run['per_role'].values())/len(run['per_role'])
            self.assertAlmostEqual(macro,run['ndcg_macro'],places=12)

    def test_chronological_split_boundaries(self):
        train,dev,fresh,_,_,_,_,_=ma.data()
        def bounds(rows):
            out={}
            for u,ctx,target,ts in rows: out.setdefault(u,[]).append(ts)
            return out
        a,b,c=map(bounds,(train,dev,fresh))
        common=set(a)&set(b)&set(c)
        self.assertGreater(len(common),1000)
        for u in common:
            self.assertLess(max(a[u]),min(b[u]))
            self.assertLess(max(b[u]),min(c[u]))

    def test_payload_roundtrip(self):
        for name in ['native','pair_gain','givens']:
            p=ROOT/'source'/f'{name}.pt'
            if not p.exists(): continue
            item=torch.load(p,map_location='cpu',weights_only=False)
            self.assertIn('state_dict',item)
            self.assertIn('item_id_map',item)
            self.assertGreater(p.stat().st_size,0)

if __name__=='__main__': unittest.main()
