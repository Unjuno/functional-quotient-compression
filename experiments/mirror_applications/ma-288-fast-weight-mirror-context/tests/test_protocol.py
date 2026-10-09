import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'source'/'run.py';sp=importlib.util.spec_from_file_location('ma288',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)

def test_heldout_rule_is_compositional():
 keys,values,rule,(ctx,idx,y),(qctx,qidx,qy)=m.make(28800,0);assert rule.shape==(m.C,m.M);assert torch.allclose(y,ctx@rule);assert torch.allclose(qy,qctx@rule)

def test_complete_context_holdout():
 _,_,_,(_,train_ids,_),(_,query_ids,_)=m.make(28800,0);assert set(train_ids.tolist()).isdisjoint(set(query_ids.tolist()))

def test_payload_charges_decoder():
 a=torch.randn(m.C,m.R);b=torch.randn(m.R,m.M);assert len(m.pack([a],'x'))<len(m.pack([a,b],'x'))
