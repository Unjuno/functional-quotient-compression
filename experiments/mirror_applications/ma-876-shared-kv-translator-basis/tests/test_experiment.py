import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from experiment import compress_maps,serialize_family,serialize_family_payloads,map_refs

class C:
    def __init__(self,layers,heads,hidden):
        self.num_hidden_layers=layers; self.num_attention_heads=heads; self.hidden_size=hidden
        self.num_key_value_heads=None


def test_relative_depth_and_head_mapping_is_deterministic():
    refs=map_refs(C(6,8,512),C(30,3,192))
    assert len(refs)==30*3*2
    assert refs[0]==(0,0,0,0,0)
    assert refs[-1]==(29,2,1,5,7)


def test_pca_map_code_reconstruction_and_serialized_basis_bytes():
    rng=np.random.default_rng(7)
    mean=rng.normal(size=12); basis=np.linalg.qr(rng.normal(size=(12,4)))[0].T
    maps=(mean+rng.normal(size=(3,4))@basis).reshape(3,3,4)
    rec,codes=compress_maps(maps,mean,basis,4)
    assert np.allclose(rec,maps,atol=1e-12)
    assert codes.shape==(3,4)


def test_affine_map_space_has_expected_dimensions():
    assert (64+1)*64==4160


def test_cached_gpt2_identity_cache_injection_preserves_next_token_logit():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from experiment import snapshot,cache_arrays,transfer_nll
    name='openai-community/gpt2'; p=snapshot(name)
    model=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,attn_implementation='eager').eval()
    tok=AutoTokenizer.from_pretrained(p,local_files_only=True)
    ids=tok('The quick brown fox jumps over the lazy dog.',add_special_tokens=False)['input_ids']
    cache,out,_=cache_arrays(model,ids)
    refs=map_refs(model.config,model.config)
    matrices=np.zeros((len(refs),65,64),dtype=np.float64)
    matrices[:,:64,:]=np.eye(64)[None,:,:]
    translated=__import__('experiment').apply_maps(cache,refs,matrices,model.config)
    plen=len(ids)-2
    translated=[(k[:,:,:plen,:].contiguous(),v[:,:,:plen,:].contiguous()) for k,v in translated]
    tnll,_=transfer_nll(model,translated,ids[-2],ids[-1])
    native=float(torch.nn.functional.cross_entropy(out.logits[0,-2].reshape(1,-1),torch.tensor([ids[-1]])))
    assert abs(tnll-native)<1e-4


def test_actual_family_payload_serialization_is_deterministic_and_byte_counted():
    maps=np.zeros((1,65,64),dtype=np.float64)
    mapset={'pair':('src/model','tgt/model'),'refs':[(0,0,0,0,0)],'maps':maps}
    mean=np.zeros(65*64); basis=np.zeros((1,65*64))
    full1,shared1=serialize_family_payloads([mapset],mean,basis,1)
    full2,shared2=serialize_family_payloads([mapset],mean,basis,1)
    assert full1==full2 and shared1==shared2
    assert serialize_family([mapset],mean,basis,1)==(len(full1),len(shared1))
