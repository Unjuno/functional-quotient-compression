import csv,hashlib,io,json,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from merger import aggregate_stats,decode,regmean_from_stats,solve_mirror_code,serialize,replay_payload

class RegMeanMirrorMathTests(unittest.TestCase):
    def test_full_basis_mirror_solution_equals_regmean_closed_form(self):
        g=torch.Generator().manual_seed(71501)
        c=torch.randn(3,5,generator=g,dtype=torch.float64)
        a=torch.randn(5,5,generator=g,dtype=torch.float64)
        gram=a.T@a/5
        w0=torch.randn(3,5,generator=g,dtype=torch.float32)
        full_basis=torch.eye(15,dtype=torch.float32).reshape(15,3,5)
        full=regmean_from_stats(gram,c,.01).float()
        code=solve_mirror_code(w0,full_basis,gram,c,.01)
        reconstructed=decode(w0,full_basis,code)
        torch.testing.assert_close(reconstructed,full,atol=2e-5,rtol=2e-5)

    def test_aggregate_stats_matches_explicit_weighted_sum(self):
        grams=[torch.eye(3,dtype=torch.float64),2*torch.eye(3,dtype=torch.float64)]
        crosses=[torch.ones(2,3,dtype=torch.float64),3*torch.ones(2,3,dtype=torch.float64)]
        alpha=torch.tensor([.25,.75])
        g,c=aggregate_stats(grams,crosses,alpha)
        torch.testing.assert_close(g,1.75*torch.eye(3,dtype=torch.float64))
        torch.testing.assert_close(c,2.5*torch.ones(2,3,dtype=torch.float64))

    def test_actual_serialized_basis_code_payload_roundtrip(self):
        w0=torch.zeros(2,3);basis=torch.eye(6).reshape(6,2,3);code=torch.arange(6.)
        expected=decode(w0,basis,code)
        path=ROOT/'source'/'_test_payload.pt'
        info=serialize({'shared':{'w0':w0,'basis':basis},'codes':code},path)
        self.assertEqual(path.stat().st_size,info['bytes'])
        self.assertEqual(replay_payload(path,expected),0.0)
        path.unlink()

    def test_actual_serialized_task_arithmetic_payload_roundtrip(self):
        w0=torch.randn(2,3,generator=torch.Generator().manual_seed(12))
        deltas=torch.randn(4,2,3,generator=torch.Generator().manual_seed(13))
        alpha=torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1]])
        source_models=w0.unsqueeze(0)+deltas
        expected=[]
        for row in alpha:expected.append(sum(float(row[i])*source_models[i] for i in range(len(row))))
        expected=torch.stack(expected)
        path=ROOT/'source'/'_test_task_payload.pt'
        info=serialize({'shared':{'source_models':source_models},'alpha':alpha},path)
        self.assertEqual(path.stat().st_size,info['bytes'])
        self.assertLessEqual(replay_payload(path,expected),5e-8)
        path.unlink()

    def test_random_draw_pool_hash_and_index_replay(self):
        with (ROOT/'source'/'selection_pool.csv').open(newline='') as f:pool=list(csv.DictReader(f))
        draw=json.loads((ROOT/'source'/'random_draw.json').read_text())
        stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(pool[0]),lineterminator='\n');w.writeheader();w.writerows(pool)
        self.assertEqual(hashlib.sha256(stream.getvalue().encode()).hexdigest(),draw['pool_csv_sha256'])
        seed=bytes.fromhex(draw['seed_hex']);limit=(1<<256)//len(pool)*len(pool);counter=0
        while True:
            value=int.from_bytes(hashlib.sha256(seed+counter.to_bytes(4,'big')).digest(),'big')
            if value<limit:break
            counter+=1
        self.assertEqual(counter,draw['rejection_counter']);idx=value%len(pool)
        self.assertEqual(idx,draw['index_zero_based']);self.assertEqual(pool[idx]['id'],'MA-715')

if __name__=='__main__':unittest.main()
