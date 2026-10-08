import hashlib,importlib.util,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma879',ROOT/'source'/'run_ma879.py');ma=importlib.util.module_from_spec(spec);spec.loader.exec_module(ma)
def test_causal_mask_blocks_future_positions():
    q=torch.ones(1,ma.L,ma.D);k=torch.ones_like(q);v=torch.eye(ma.L).unsqueeze(-1).expand(-1,-1,ma.D).unsqueeze(0).float();p,_=ma.attend(q,k,v)
    assert torch.allclose(p[0].triu(1),torch.zeros_like(p[0].triu(1)))
def test_methods_forward_and_gradient():
    sk=torch.randn(3,ma.L,ma.D);sv=torch.randn_like(sk);t=torch.tensor([0,1,2])
    for method in ma.METHODS:
        model=ma.Translator(method);k,v=model(sk,sv,t);assert k.shape==sv.shape and v.shape==sv.shape;(k.square().mean()+v.square().mean()).backward();assert any(p.grad is not None for p in model.parameters())
def test_aligned_teacher_is_deterministic():
    for variant in ma.VARIANTS:
        k,v=ma.teacher(11,variant);k2,v2=ma.teacher(11,variant);assert torch.equal(torch.tensor(k),torch.tensor(k2)) and torch.equal(torch.tensor(v),torch.tensor(v2))
def test_draw32_replay():
    d=json.loads((ROOT/'source'/'draw32_exclusions.json').read_text());pool=d['pool_ids'];ph=bytes.fromhex(d['pool_sha256']);seed=bytes.fromhex(d['seed_hex']);h=hashlib.sha256(seed+ph+d['rejection_counter'].to_bytes(4,'big')).digest();n=len(pool);v=int.from_bytes(h,'big');lim=(1<<256)-((1<<256)%n)
    assert len(pool)==d['eligible_count'] and hashlib.sha256('\n'.join(pool).encode()).hexdigest()==d['pool_sha256'];assert v<lim and v%n==d['selection_index_zero_based'];assert pool[v%n]=='MA-879'
