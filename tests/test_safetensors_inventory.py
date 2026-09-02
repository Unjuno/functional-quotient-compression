import json
from pathlib import Path

import pytest

from fqc.hf_llama_adapter import build_hf_llama_adapter_plan
from fqc.safetensors_inventory import (
    materialize_adapter_plan_from_safetensors_header,
    read_safetensors_header,
    sha256_file,
)


def _write_safetensors(path: Path, tensors):
    header={}; offset=0; chunks=[]
    for key,dtype,shape in tensors:
        n=1
        for d in shape:n*=d
        item={'BF16':2,'F32':4}[dtype]
        size=n*item
        header[key]={'dtype':dtype,'shape':list(shape),'data_offsets':[offset,offset+size]}
        chunks.append(bytes(size)); offset+=size
    raw=json.dumps(header,separators=(',',':')).encode()
    path.write_bytes(len(raw).to_bytes(8,'little')+raw+b''.join(chunks))


def _tiny_tensors():
    out=[('model.embed_tokens.weight','BF16',(10,4))]
    for i in range(2):
        p=f'model.layers.{i}'
        out += [
            (f'{p}.self_attn.q_proj.weight','BF16',(4,4)),
            (f'{p}.self_attn.k_proj.weight','BF16',(2,4)),
            (f'{p}.self_attn.v_proj.weight','BF16',(2,4)),
            (f'{p}.self_attn.o_proj.weight','BF16',(4,4)),
            (f'{p}.mlp.gate_proj.weight','BF16',(6,4)),
            (f'{p}.mlp.up_proj.weight','BF16',(6,4)),
            (f'{p}.mlp.down_proj.weight','BF16',(4,6)),
            (f'{p}.input_layernorm.weight','BF16',(4,)),
            (f'{p}.post_attention_layernorm.weight','BF16',(4,)),
        ]
    out.append(('model.norm.weight','BF16',(4,)))
    return out


def _tiny_config():
    return {'model_type':'llama','vocab_size':10,'hidden_size':4,'intermediate_size':6,'num_hidden_layers':2,
            'num_attention_heads':2,'num_key_value_heads':1,'hidden_act':'silu','tie_word_embeddings':True,
            'rope_theta':10000,'rope_scaling':None,'rms_norm_eps':1e-5,'attention_bias':False,'mlp_bias':False}


def test_header_parser_counts_real_serialized_scalars_and_bytes(tmp_path):
    p=tmp_path/'model.safetensors'; _write_safetensors(p,_tiny_tensors())
    h=read_safetensors_header(p)
    assert len(h.entries)==20
    assert h.scalar_count==300
    assert sum(x.data_bytes for x in h.entries.values())==600
    assert len(sha256_file(p))==64


def test_serialized_header_materializes_complete_tied_manifest(tmp_path):
    p=tmp_path/'model.safetensors'; _write_safetensors(p,_tiny_tensors())
    h=read_safetensors_header(p)
    plan=build_hf_llama_adapter_plan(_tiny_config(),checkpoint_sha256='sha256:abc',model_id='tiny',available_checkpoint_keys=h.entries)
    r=materialize_adapter_plan_from_safetensors_header(plan,h)
    assert r.passed
    assert r.serialized_scalar_count==300==r.config_expected_scalar_count
    assert r.serialized_tensor_count==20
    assert r.dtype_scalar_counts=={'BF16':300}
    groups={x['tensor_id']:x['storage_group'] for x in r.extraction_manifest['tensor_inventory']}
    assert groups['tok_emb']==groups['lm_head']


def test_unaccounted_checkpoint_tensor_fails_closed(tmp_path):
    p=tmp_path/'model.safetensors'; _write_safetensors(p,_tiny_tensors()+[('mystery.weight','BF16',(1,))])
    h=read_safetensors_header(p)
    plan=build_hf_llama_adapter_plan(_tiny_config(),checkpoint_sha256='sha256:abc',model_id='tiny',available_checkpoint_keys=h.entries)
    with pytest.raises(ValueError,match='unaccounted tensor keys'):
        materialize_adapter_plan_from_safetensors_header(plan,h)


def test_bad_tensor_byte_range_is_rejected(tmp_path):
    p=tmp_path/'bad.safetensors'
    header={'x':{'dtype':'BF16','shape':[2],'data_offsets':[0,3]}}
    raw=json.dumps(header,separators=(',',':')).encode(); p.write_bytes(len(raw).to_bytes(8,'little')+raw+b'abc')
    with pytest.raises(ValueError,match='data byte count'):
        read_safetensors_header(p)
