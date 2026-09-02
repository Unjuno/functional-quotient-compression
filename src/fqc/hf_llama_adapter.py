"""Explicit Hugging Face Llama-family adapter-plan construction.

This module uses declared config fields and known Transformers Llama module
structure. It does not inspect tensor values or infer architecture from shapes.
"""
from __future__ import annotations

from typing import Any, Collection, Mapping

from .adapter_plan import AdapterPlan, TensorBinding
from .linear_orientation import PYTORCH_OUT_IN, ROW_VECTOR_X_W
from .manifest_builder import contiguous_gqa_map


def _positive_int(config: Mapping[str, Any], key: str) -> int:
    value=config.get(key)
    if not isinstance(value,int) or isinstance(value,bool) or value<=0:
        raise ValueError(f'{key} must be a positive integer')
    return value


def _geometry(config: Mapping[str, Any]) -> tuple[int,int,int,int,int,int]:
    d_model=_positive_int(config,'hidden_size')
    intermediate=_positive_int(config,'intermediate_size')
    layers=_positive_int(config,'num_hidden_layers')
    q_heads=_positive_int(config,'num_attention_heads')
    kv_heads=_positive_int(config,'num_key_value_heads')
    head_dim=config.get('head_dim')
    if head_dim is None:
        if d_model%q_heads:
            raise ValueError('hidden_size must be divisible by num_attention_heads when head_dim is absent')
        head_dim=d_model//q_heads
    if not isinstance(head_dim,int) or head_dim<=0:
        raise ValueError('head_dim must be a positive integer')
    if q_heads*head_dim!=d_model:
        raise ValueError('current FQC Llama adapter requires q_heads * head_dim == hidden_size')
    return d_model,intermediate,layers,q_heads,kv_heads,head_dim


def _check_supported_biases(config: Mapping[str, Any]) -> None:
    if bool(config.get('attention_bias',False)):
        raise ValueError('attention_bias=true is not supported by the current FQC Llama adapter')
    if bool(config.get('mlp_bias',False)):
        raise ValueError('mlp_bias=true is not supported by the current FQC Llama adapter')


def expected_hf_llama_unique_scalars(config: Mapping[str, Any]) -> int:
    """Config-derived scalar-count cross-check for the currently supported Llama layout.

    This is not authoritative for a real compression denominator; live storage
    inventory remains authoritative. It is used to catch adapter/storage drift.
    """
    if config.get('model_type')!='llama':
        raise ValueError('model_type must be llama')
    _check_supported_biases(config)
    d_model,intermediate,layers,q_heads,kv_heads,head_dim=_geometry(config)
    vocab=_positive_int(config,'vocab_size')
    tie=bool(config.get('tie_word_embeddings',False))
    embedding=vocab*d_model
    lm_head=0 if tie else vocab*d_model
    attention=(d_model*q_heads*head_dim + d_model*kv_heads*head_dim +
               d_model*kv_heads*head_dim + q_heads*head_dim*d_model)
    mlp=3*d_model*intermediate
    per_layer=attention+mlp+2*d_model
    final_norm=d_model
    return embedding+lm_head+layers*per_layer+final_norm


def build_hf_llama_adapter_plan(
    config: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    model_id: str,
    available_checkpoint_keys: Collection[str] | None = None,
    adapter_version: str = '1',
) -> AdapterPlan:
    """Build an explicit plan for Transformers `LlamaForCausalLM` checkpoints."""
    if config.get('model_type')!='llama':
        raise ValueError('model_type must be llama')
    if not checkpoint_sha256.startswith('sha256:'):
        raise ValueError('checkpoint_sha256 must use sha256:<hex> form')
    _check_supported_biases(config)
    d_model,intermediate,layers,q_heads,kv_heads,head_dim=_geometry(config)
    q_to_kv=contiguous_gqa_map(q_heads,kv_heads)
    tie=bool(config.get('tie_word_embeddings',False))
    hidden_act=config.get('hidden_act')
    if hidden_act not in ('silu','swiglu'):
        raise ValueError(f'unsupported Llama hidden_act for current adapter: {hidden_act}')

    keys=set(available_checkpoint_keys) if available_checkpoint_keys is not None else None
    embed_key='model.embed_tokens.weight'; lm_key='lm_head.weight'
    if keys is not None:
        if embed_key not in keys: raise ValueError(f'missing checkpoint key: {embed_key}')
        if tie and lm_key not in keys:
            lm_key=embed_key
        elif lm_key not in keys:
            raise ValueError(f'missing checkpoint key: {lm_key}')

    bindings=[
        TensorBinding('tok_emb',embed_key,'token_embedding'),
        TensorBinding('lm_head',lm_key,'lm_head'),
    ]
    attention=[]; mlps=[]; norms=[]; primitives=[]
    for i in range(layers):
        p=f'model.layers.{i}'
        public=f'L{i}'
        layer_bindings=[
            ('WQ',f'{p}.self_attn.q_proj.weight','attention_q'),
            ('WK',f'{p}.self_attn.k_proj.weight','attention_k'),
            ('WV',f'{p}.self_attn.v_proj.weight','attention_v'),
            ('WO',f'{p}.self_attn.o_proj.weight','attention_o'),
            ('gate',f'{p}.mlp.gate_proj.weight','mlp_gate'),
            ('up',f'{p}.mlp.up_proj.weight','mlp_up'),
            ('down',f'{p}.mlp.down_proj.weight','mlp_down'),
            ('input_norm',f'{p}.input_layernorm.weight','norm_scale'),
            ('post_attn_norm',f'{p}.post_attention_layernorm.weight','norm_scale'),
        ]
        for suffix,key,role in layer_bindings:
            if keys is not None and key not in keys: raise ValueError(f'missing checkpoint key: {key}')
            bindings.append(TensorBinding(f'{public}.{suffix}',key,role))
        attention.append({
            'module_id':f'{public}.attn','d_model':d_model,'q_heads':q_heads,'kv_heads':kv_heads,
            'head_dim':head_dim,'value_dim':head_dim,'q_to_kv_map':q_to_kv,
            'projection_tensor_ids':{'WQ':f'{public}.WQ','WK':f'{public}.WK','WV':f'{public}.WV','WO':f'{public}.WO'},
            'position_operator':'rope','rope':{'rotary_dim':head_dim,'frequency_source':'EXTERNAL_FIXED','layout':'pairwise_2d'},
            'qk_norm':'none','bias':{'q':False,'k':False,'v':False,'o':False},
            'logit_scale':'1/sqrt(head_dim)','output_block_policy':'split_WO_by_query_head',
            'exact_reduction_policy':'rope_profile_operator_family',
        })
        mlps.append({
            'module_id':f'{public}.mlp','mlp_type':'swiglu','activation':'silu',
            'tensor_ids':{'gate':f'{public}.gate','up':f'{public}.up','down':f'{public}.down'},
            'symmetry_policy':{'hidden_permutation':'exact','up_branch_scaling_with_inverse_down':'exact','gate_branch_scaling':'not_claimed'},
        })
        norms.extend([
            {'module_id':f'{public}.input_norm','type':'rmsnorm','tensor_ids':[f'{public}.input_norm'],'analysis_policy':'retain_exact_tensor'},
            {'module_id':f'{public}.post_attn_norm','type':'rmsnorm','tensor_ids':[f'{public}.post_attn_norm'],'analysis_policy':'retain_exact_tensor'},
        ])
        primitives.extend([
            {'primitive_id':f'{public}.attn.rope_family','kind':'rope_A_B_pair_family','source_tensor_ids':[f'{public}.WQ',f'{public}.WK'],'analysis_only':True,'paid_if_serialized':True},
            {'primitive_id':f'{public}.attn.value_output','kind':'value_output_R','source_tensor_ids':[f'{public}.WV',f'{public}.WO'],'analysis_only':True,'paid_if_serialized':True},
        ])
    final_norm='model.norm.weight'
    if keys is not None and final_norm not in keys: raise ValueError(f'missing checkpoint key: {final_norm}')
    bindings.append(TensorBinding('final_norm',final_norm,'norm_scale'))
    norms.append({'module_id':'final_norm','type':'rmsnorm','tensor_ids':['final_norm'],'analysis_policy':'retain_exact_tensor'})

    config_evidence={
        'model_type':'llama','hidden_size':d_model,'intermediate_size':intermediate,
        'num_hidden_layers':layers,'num_attention_heads':q_heads,'num_key_value_heads':kv_heads,
        'head_dim':head_dim,'hidden_act':hidden_act,'tie_word_embeddings':tie,
        'rope_theta':config.get('rope_theta'),'rope_scaling':config.get('rope_scaling'),
        'rms_norm_eps':config.get('rms_norm_eps'),'attention_bias':False,'mlp_bias':False,
        'config_expected_unique_scalar_count':expected_hf_llama_unique_scalars(config),
    }
    return AdapterPlan(
        adapter_id='hf-transformers-llama',adapter_version=adapter_version,
        model_identity={
            'model_id':model_id,'architecture_family':'LlamaForCausalLM','checkpoint_sha256':checkpoint_sha256,
            'checkpoint_weight_orientation':PYTORCH_OUT_IN,'canonical_operator_orientation':ROW_VECTOR_X_W,
        },
        tensor_bindings=bindings,attention_modules=attention,mlp_modules=mlps,normalization_modules=norms,
        external_fixed_state=['causal_mask_rule','RoPE_frequency_rule_if_declared_external'],
        derived_primitives=primitives,
        extraction_policy={
            'residual_stream_basis':'retain_unless_cross_module_proof',
            'embedding_lm_head_ties':'count_storage_once_keep_roles',
            'analysis_primitive_zero_bit_assumption':'forbidden',
        },
        config_evidence=config_evidence,
    )
