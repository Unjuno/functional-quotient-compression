#!/usr/bin/env python3
"""Run T001 SmolLM2 live-storage consistency and layer-0 replay."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from fqc.adapter_plan import materialize_adapter_plan
from fqc.hf_llama_adapter import build_hf_llama_adapter_plan
from fqc.llama_runtime_replay import (
    default_llama_rope,
    manual_llama_attention,
    manual_llama_decoder_layer,
    manual_llama_mlp,
    rms_norm,
)
from fqc.manifest_builder import sha256_json
from fqc.replay import ReplayCase, build_replay_witness
from fqc.safetensors_inventory import read_safetensors_header, sha256_file


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    return np.asarray(x)


def _same_tensor_storage(a: torch.Tensor,b: torch.Tensor) -> bool:
    sa=a.untyped_storage(); sb=b.untyped_storage()
    return (
        int(sa.data_ptr())==int(sb.data_ptr()) and
        int(a.storage_offset())==int(b.storage_offset()) and
        int(a.numel())==int(b.numel()) and
        int(a.element_size())==int(b.element_size())
    )


def _causal_mask(seq_len: int, *, dtype, device):
    mask=torch.zeros((1,1,seq_len,seq_len),dtype=dtype,device=device)
    upper=torch.triu(torch.ones((seq_len,seq_len),dtype=torch.bool,device=device),diagonal=1)
    return mask.masked_fill(upper,torch.finfo(dtype).min)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--pin',default='experiments/transformer/T001_smollm2_135m/pilot_pin.json')
    ap.add_argument('--replay-contract',default='experiments/transformer/T001_smollm2_135m/runtime_replay_contract.json')
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()

    model_dir=Path(args.model_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    pin=json.loads(Path(args.pin).read_text()); contract=json.loads(Path(args.replay_contract).read_text())
    checkpoint=model_dir/'model.safetensors'
    actual_sha=sha256_file(checkpoint)
    if actual_sha!=pin['checkpoint_file']['sha256']:
        raise SystemExit(f'checkpoint SHA mismatch: {actual_sha}')
    if transformers.__version__!=contract['transformers_version']:
        raise SystemExit(f'transformers version mismatch: {transformers.__version__}')
    if torch.__version__!=contract['torch_version']:
        raise SystemExit(f'torch version mismatch: {torch.__version__}')

    header=read_safetensors_header(checkpoint)
    plan=build_hf_llama_adapter_plan(
        pin['config'],checkpoint_sha256='sha256:'+actual_sha,model_id=pin['model_id'],
        available_checkpoint_keys=header.entries,
    )

    torch.manual_seed(int(contract['seed']))
    model=AutoModelForCausalLM.from_pretrained(
        str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation=contract['attention_implementation'],
    )
    model.eval()
    state=model.state_dict(keep_vars=True)
    source_keys=set(header.entries)
    runtime_keys=set(state)
    missing_runtime=tuple(sorted(source_keys-runtime_keys))
    if missing_runtime:
        raise SystemExit('runtime state is missing source checkpoint keys: '+', '.join(missing_runtime))

    runtime_checkpoint={k:state[k] for k in source_keys}
    runtime_manifest=materialize_adapter_plan(plan,runtime_checkpoint)
    runtime_manifest['model_identity']['hub_revision']=pin['hub_revision']
    runtime_manifest['model_identity']['checkpoint_file']=pin['checkpoint_file']['path']
    runtime_manifest['model_identity']['runtime_transformers_version']=transformers.__version__
    runtime_manifest['model_identity']['runtime_torch_version']=torch.__version__
    runtime_manifest_hash=sha256_json(runtime_manifest)
    runtime_n=runtime_manifest['model_identity']['unique_baseline_scalar_count_N']
    source_n=pin['authority']['denominator_N']

    embedding=model.get_input_embeddings().weight
    lm_head=model.get_output_embeddings().weight
    tied_same_object=embedding is lm_head
    tied_same_storage=_same_tensor_storage(embedding,lm_head)

    layer_index=int(contract['layer_index'])
    layer=model.model.layers[layer_index]
    cfg=pin['config']; batch=int(contract['batch_size']); seqlen=int(contract['sequence_length'])
    hidden=torch.randn((batch,seqlen,cfg['hidden_size']),dtype=torch.float32).to(torch.bfloat16)
    position_ids=torch.arange(seqlen,dtype=torch.long).unsqueeze(0).expand(batch,-1)
    attention_mask=_causal_mask(seqlen,dtype=hidden.dtype,device=hidden.device)

    manual_cos,manual_sin=default_llama_rope(
        position_ids,head_dim=cfg['head_dim'],theta=cfg['rope_theta'],dtype=hidden.dtype,device=hidden.device,
    )
    hf_cos,hf_sin=model.model.rotary_emb(hidden,position_ids)

    p=f'model.layers.{layer_index}'
    weights={
        'WQ':state[f'{p}.self_attn.q_proj.weight'].detach(),
        'WK':state[f'{p}.self_attn.k_proj.weight'].detach(),
        'WV':state[f'{p}.self_attn.v_proj.weight'].detach(),
        'WO':state[f'{p}.self_attn.o_proj.weight'].detach(),
        'gate':state[f'{p}.mlp.gate_proj.weight'].detach(),
        'up':state[f'{p}.mlp.up_proj.weight'].detach(),
        'down':state[f'{p}.mlp.down_proj.weight'].detach(),
        'input_norm':state[f'{p}.input_layernorm.weight'].detach(),
        'post_attn_norm':state[f'{p}.post_attention_layernorm.weight'].detach(),
    }

    with torch.no_grad():
        ref_norm=layer.input_layernorm(hidden)
        got_norm=rms_norm(hidden,weights['input_norm'],cfg['rms_norm_eps'])

        ref_attn=layer.self_attn(
            hidden_states=ref_norm,attention_mask=attention_mask,position_ids=position_ids,
            use_cache=False,output_attentions=False,position_embeddings=(hf_cos,hf_sin),
        )[0]
        got_attn=manual_llama_attention(
            got_norm,weights,q_heads=cfg['num_attention_heads'],kv_heads=cfg['num_key_value_heads'],
            head_dim=cfg['head_dim'],attention_mask=attention_mask,cos=manual_cos,sin=manual_sin,
        )

        mlp_probe=torch.randn((batch,seqlen,cfg['hidden_size']),dtype=torch.float32).to(torch.bfloat16)
        ref_mlp=layer.mlp(mlp_probe)
        got_mlp=manual_llama_mlp(mlp_probe,weights)

        ref_full=layer(
            hidden_states=hidden,attention_mask=attention_mask,position_ids=position_ids,
            use_cache=False,output_attentions=False,position_embeddings=(hf_cos,hf_sin),
        )[0]
        got_full=manual_llama_decoder_layer(
            hidden,weights,q_heads=cfg['num_attention_heads'],kv_heads=cfg['num_key_value_heads'],
            head_dim=cfg['head_dim'],rms_norm_eps=cfg['rms_norm_eps'],attention_mask=attention_mask,
            cos=manual_cos,sin=manual_sin,
        )

    cases=[
        ReplayCase('input_rms_norm',{'hidden':_np(hidden)},_np(ref_norm),_np(got_norm)),
        ReplayCase('rope_cos_sin',{'position_ids':_np(position_ids)},np.concatenate([_np(hf_cos),_np(hf_sin)],axis=-1),np.concatenate([_np(manual_cos),_np(manual_sin)],axis=-1)),
        ReplayCase('attention_output',{'hidden':_np(hidden),'position_ids':_np(position_ids)},_np(ref_attn),_np(got_attn)),
        ReplayCase('mlp_output',{'mlp_probe':_np(mlp_probe)},_np(ref_mlp),_np(got_mlp)),
        ReplayCase('full_decoder_layer',{'hidden':_np(hidden),'position_ids':_np(position_ids)},_np(ref_full),_np(got_full)),
    ]
    witness=build_replay_witness(cases,contract)
    result={
        'experiment_id':pin['experiment_id'],
        'status':'PASS',
        'evidence_scope':'RUNTIME_STORAGE_AND_LAYER0_REPLAY_ONLY_NO_COMPRESSION_RESULT',
        'hub_revision':pin['hub_revision'],
        'checkpoint_sha256':actual_sha,
        'source_denominator_N':source_n,
        'runtime_manifest_unique_scalar_count_N':runtime_n,
        'runtime_manifest_canonical_sha256':runtime_manifest_hash,
        'runtime_source_keys_missing':list(missing_runtime),
        'runtime_state_extra_keys':sorted(runtime_keys-source_keys),
        'embedding_lm_head_same_parameter_object':tied_same_object,
        'embedding_lm_head_same_storage':tied_same_storage,
        'adapter_version':plan.adapter_version,
        'rope_contract':plan.attention_modules[layer_index]['rope'],
        'transformers_version':transformers.__version__,
        'torch_version':torch.__version__,
        'attention_implementation':contract['attention_implementation'],
        'replay_contract_sha256':witness.contract_hash,
        'replay_passed':witness.passed,
        'replay_cases':[asdict(x) for x in witness.cases],
        'runner_provenance':{
            'github_run_id':os.environ.get('GITHUB_RUN_ID'),
            'github_sha':os.environ.get('GITHUB_SHA'),
            'github_ref':os.environ.get('GITHUB_REF'),
            'github_workflow':os.environ.get('GITHUB_WORKFLOW'),
        },
    }
    passed=(runtime_n==source_n and tied_same_storage and witness.passed and not missing_runtime)
    result['status']='PASS' if passed else 'FAIL'
    (out/'runtime_replay_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'runtime_manifest.json').write_text(json.dumps(runtime_manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
