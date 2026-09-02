from fqc.adapter_plan import materialize_adapter_plan
from fqc.hf_llama_adapter import build_hf_llama_adapter_plan


class _Storage:
    def __init__(self,ptr,nbytes):self.p=ptr;self.n=nbytes
    def data_ptr(self):return self.p
    def nbytes(self):return self.n
class _Tensor:
    device='cpu';dtype='bf16'
    def __init__(self,shape,ptr):
        self.shape=shape; self._n=1
        for d in shape:self._n*=d
        self.s=_Storage(ptr,2*self._n)
    def is_contiguous(self):return True
    def untyped_storage(self):return self.s
    def element_size(self):return 2
    def storage_offset(self):return 0
    def numel(self):return self._n

CONFIG={'model_type':'llama','hidden_size':4,'intermediate_size':6,'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'hidden_act':'silu','tie_word_embeddings':True,'rope_theta':10000,'rope_scaling':None,'rms_norm_eps':1e-5,'attention_bias':False}


def _checkpoint():
    ck={}; ptr=100
    emb=_Tensor((10,4),ptr); ptr+=1
    ck['model.embed_tokens.weight']=emb
    # Deliberately omit lm_head.weight: tied config should bind both roles to embedding storage.
    for i in range(2):
        p=f'model.layers.{i}'
        for key,shape in [
            ('self_attn.q_proj.weight',(4,4)),('self_attn.k_proj.weight',(2,4)),('self_attn.v_proj.weight',(2,4)),('self_attn.o_proj.weight',(4,4)),
            ('mlp.gate_proj.weight',(6,4)),('mlp.up_proj.weight',(6,4)),('mlp.down_proj.weight',(4,6)),
            ('input_layernorm.weight',(4,)),('post_attention_layernorm.weight',(4,)),
        ]:
            ck[f'{p}.{key}']=_Tensor(shape,ptr); ptr+=1
    ck['model.norm.weight']=_Tensor((4,),ptr)
    return ck


def test_llama_adapter_materializes_pytorch_shapes_and_tied_embedding():
    ck=_checkpoint()
    plan=build_hf_llama_adapter_plan(CONFIG,checkpoint_sha256='sha256:abc',model_id='tiny',available_checkpoint_keys=ck)
    manifest=materialize_adapter_plan(plan,ck)
    assert manifest['model_identity']['checkpoint_weight_orientation']=='pytorch_linear_weight_out_in'
    assert manifest['model_identity']['unique_baseline_scalar_count_N']==300
    groups={x['tensor_id']:x['storage_group'] for x in manifest['tensor_inventory']}
    assert groups['tok_emb']==groups['lm_head']
    assert len(manifest['modules']['attention'])==2
    assert manifest['modules']['attention'][0]['q_to_kv_map']==[0,0]


def test_smollm2_135m_config_generates_expected_geometry_without_weights():
    config={'model_type':'llama','hidden_size':576,'intermediate_size':1536,'num_hidden_layers':30,'num_attention_heads':9,'num_key_value_heads':3,'hidden_act':'silu','tie_word_embeddings':True,'rope_theta':100000,'rope_scaling':None,'rms_norm_eps':1e-5,'attention_bias':False}
    plan=build_hf_llama_adapter_plan(config,checkpoint_sha256='sha256:placeholder',model_id='HuggingFaceTB/SmolLM2-135M')
    a=plan.attention_modules[0]
    assert a['head_dim']==64 and a['q_heads']==9 and a['kv_heads']==3
    assert a['q_to_kv_map']==[0,0,0,1,1,1,2,2,2]
    assert len(plan.attention_modules)==30
