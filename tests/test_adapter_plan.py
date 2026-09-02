import pytest

from fqc.adapter_plan import AdapterPlan, TensorBinding, adapter_plan_payload, materialize_adapter_plan
from fqc.manifest_builder import canonical_json_bytes, contiguous_gqa_map, sha256_json


class _Storage:
    def __init__(self, ptr,nbytes): self.ptr=ptr; self.nb=nbytes
    def data_ptr(self): return self.ptr
    def nbytes(self): return self.nb
class _Tensor:
    device='cpu'; dtype='bf16'
    def __init__(self,shape,ptr): self.shape=shape; self.s=_Storage(ptr,2*self.numel_from(shape))
    @staticmethod
    def numel_from(shape):
        n=1
        for d in shape:n*=d
        return n
    def is_contiguous(self): return True
    def untyped_storage(self): return self.s
    def element_size(self): return 2
    def storage_offset(self): return 0
    def numel(self): return self.numel_from(self.shape)


def _plan():
    bindings=[
        TensorBinding('tok','model.embed_tokens.weight','token_embedding'),
        TensorBinding('head','lm_head.weight','lm_head'),
        TensorBinding('q','layers.0.q_proj.weight','attention_q'),
        TensorBinding('k','layers.0.k_proj.weight','attention_k'),
        TensorBinding('v','layers.0.v_proj.weight','attention_v'),
        TensorBinding('o','layers.0.o_proj.weight','attention_o'),
    ]
    attn=[{
        'module_id':'L0.attn','d_model':4,'q_heads':2,'kv_heads':1,'head_dim':2,'value_dim':2,
        'q_to_kv_map':contiguous_gqa_map(2,1),
        'projection_tensor_ids':{'WQ':'q','WK':'k','WV':'v','WO':'o'},
        'position_operator':'rope','rope':{'rotary_dim':2},'qk_norm':'none',
        'exact_reduction_policy':'rope_profile_operator_family',
    }]
    return AdapterPlan(
        adapter_id='explicit-test',adapter_version='1',
        model_identity={'architecture_family':'llama_like_test','orientation':'row_vector_x_times_W','checkpoint_sha256':'sha256:abc'},
        tensor_bindings=bindings,attention_modules=attn,
        config_evidence={'hidden_size':4,'num_attention_heads':2,'num_key_value_heads':1,'rope':True},
    )


def _checkpoint(ptr=100):
    tok=_Tensor((10,4),ptr)
    return {
        'model.embed_tokens.weight':tok,
        'lm_head.weight':tok,
        'layers.0.q_proj.weight':_Tensor((4,4),ptr+1),
        'layers.0.k_proj.weight':_Tensor((4,2),ptr+2),
        'layers.0.v_proj.weight':_Tensor((4,2),ptr+3),
        'layers.0.o_proj.weight':_Tensor((4,4),ptr+4),
    }


def test_materialized_plan_is_valid_and_tracks_binding_provenance():
    m=materialize_adapter_plan(_plan(),_checkpoint())
    assert m['model_identity']['unique_baseline_scalar_count_N']==88
    adapter=m['model_identity']['adapter']
    assert adapter['adapter_id']=='explicit-test'
    assert adapter['adapter_plan_sha256']==sha256_json(adapter_plan_payload(_plan()))
    mapping={x['tensor_id']:x['checkpoint_key'] for x in m['extraction_policy']['tensor_binding_provenance']}
    assert mapping['q']=='layers.0.q_proj.weight'


def test_adapter_manifest_is_pointer_independent():
    a=materialize_adapter_plan(_plan(),_checkpoint(100))
    b=materialize_adapter_plan(_plan(),_checkpoint(900))
    assert canonical_json_bytes(a)==canonical_json_bytes(b)


def test_missing_checkpoint_key_is_rejected():
    ckpt=_checkpoint(); del ckpt['layers.0.k_proj.weight']
    with pytest.raises(ValueError,match='missing checkpoint tensors'):
        materialize_adapter_plan(_plan(),ckpt)


def test_duplicate_public_ids_are_rejected():
    p=_plan()
    bindings=list(p.tensor_bindings)+[TensorBinding('q','extra','other')]
    bad=AdapterPlan(**{**p.__dict__,'tensor_bindings':bindings})
    with pytest.raises(ValueError,match='public_id'):
        materialize_adapter_plan(bad,{**_checkpoint(),'extra':_Tensor((1,),999)})
