import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

PATH=Path(__file__).parents[1]/'source'/'experiment.py'
spec=importlib.util.spec_from_file_location('ma882',PATH); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def test_source_anchors_use_normalized_depth():
    assert m.source_anchors(SimpleNamespace(num_hidden_layers=6))==(0,2,5)
    assert m.source_anchors(SimpleNamespace(num_hidden_layers=12))==(0,6,11)

def test_routes_pack_only_selected_maps():
    maps=np.zeros((3,30,3,2,65,64),np.float32)
    mapsets=[{'maps':maps,'pair':('s0','t')},{'maps':maps,'pair':('s1','t')}]
    routes=[np.tile(np.array([[0,1],[1,2]],np.int8),(15,1)) for _ in mapsets]
    bank,mask,ids=m.pack_selected(mapsets,routes)
    assert bank.shape==(2,3,30,3,2,65,64)
    assert mask.sum()==2*30*2*6
    assert len(ids)==mask.sum()

def test_cp_outer_product_reconstructs_exactly():
    factors=[torch.tensor([[1.],[2.]]),torch.tensor([[3.],[4.],[5.]]),torch.tensor([[6.]]),torch.tensor([[7.],[8.]]),torch.tensor([[9.]])]
    out=m.cp_reconstruct(factors)
    assert out.shape==(2,3,1,2,1)
    assert out[1,2,0,1,0].item()==2*5*6*8*9

def test_fixed_route_map_application_uses_all_target_layers():
    source_cfg=SimpleNamespace(num_hidden_layers=3,model_type='gpt2',num_key_value_heads=3,num_attention_heads=3)
    target_cfg=SimpleNamespace(num_hidden_layers=2,model_type='gpt2',num_key_value_heads=3,num_attention_heads=3)
    source_cache=[]
    for layer in range(3):
        k=np.full((3,4,64),layer+1,dtype=np.float64)
        v=np.full((3,4,64),10+layer,dtype=np.float64)
        source_cache.append((k,v))
    bank=np.zeros((2,3,2,3,2,65,64),np.float32)
    for p in range(2):
      for a in range(3):
       for t in range(2):
        for h in range(3):
         for kv in range(2): bank[p,a,t,h,kv,:64,:]=np.eye(64,dtype=np.float32)
    routes=np.array([[0,1],[1,2]],np.int8)
    out=m.apply_bank(source_cache,bank,0,routes,source_cfg,target_cfg,4)
    assert len(out)==2 and out[0][0].shape==(1,3,4,64)
    assert torch.allclose(out[0][0],torch.full((1,3,4,64),1.5))
    assert torch.allclose(out[1][1],torch.full((1,3,4,64),11.5))

def test_all_serialized_inference_payloads_roundtrip_exact_zero_maps():
    mapsets=[{'maps':np.zeros((3,30,3,2,65,64),np.float32)} for _ in range(2)]
    routes=[np.tile(np.array([[0,1],[1,2]],np.int8),(15,1)) for _ in mapsets]
    bank,_,ids=m.pack_selected(mapsets,routes)
    mean=np.zeros((65*64,),np.float32); basis=np.zeros((32,65*64),np.float32); codes=np.zeros((len(ids),32),np.float32)
    raws=[m.serialize_full(bank,ids,routes),m.serialize_pca(mean,basis,codes,32,ids,routes)]
    factors=[np.zeros((2,2),np.float32),np.zeros((3,2),np.float32),np.zeros((30,2),np.float32),np.zeros((6,2),np.float32),np.zeros((32,2),np.float32)]
    raws.append(m.serialize_cp(mean,basis,factors,32,2,ids,routes))
    for raw,want in zip(raws,('full','pca','cp')):
        name,decoded,rte=m.decode_payload(raw)
        assert name==want and np.array_equal(decoded,bank)
        assert all(np.array_equal(a,b) for a,b in zip(routes,rte))
