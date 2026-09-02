#!/usr/bin/env python3
"""Run T004: separate local Q/K intervention effects from downstream amplification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fqc.continuation_metrics import pearson_correlation, relative_rms_delta, safe_ratio
from fqc.functional_interventions import normalized_family_rank_approximation
from fqc.manifest_builder import sha256_json
from fqc.safetensors_inventory import sha256_file


def _tensor_hash(t: torch.Tensor) -> str:
    a=t.detach().cpu().contiguous().numpy()
    h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(repr(tuple(a.shape)).encode()); h.update(a.tobytes())
    return 'sha256:'+h.hexdigest()


def _tokenize(tokenizer, calibration):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None: raise RuntimeError('tokenizer has neither pad nor eos token')
        tokenizer.pad_token=tokenizer.eos_token
    batch=tokenizer(
        calibration['texts'],return_tensors='pt',padding=True,truncation=True,
        max_length=int(calibration['max_length']),add_special_tokens=True,
    )
    return batch['input_ids'],batch['attention_mask']


def _first_tensor(output):
    if isinstance(output,torch.Tensor): return output
    if isinstance(output,(tuple,list)) and output and isinstance(output[0],torch.Tensor): return output[0]
    raise RuntimeError('hook output does not expose a tensor in position 0')


def _forward_capture(model,input_ids,attention_mask,layers):
    captured={int(i):{} for i in layers}; handles=[]
    for i in layers:
        layer=model.model.layers[int(i)]
        def attn_hook(_module,_inputs,output,layer_id=int(i)):
            captured[layer_id]['attention']=_first_tensor(output).detach().float().cpu()
        def layer_hook(_module,_inputs,output,layer_id=int(i)):
            captured[layer_id]['decoder']=_first_tensor(output).detach().float().cpu()
        handles.append(layer.self_attn.register_forward_hook(attn_hook))
        handles.append(layer.register_forward_hook(layer_hook))
    try:
        with torch.no_grad():
            logits=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).logits.float().cpu()
    finally:
        for h in handles: h.remove()
    for i in layers:
        if set(captured[int(i)])!={'attention','decoder'}:
            raise RuntimeError(f'missing captures for layer {i}: {captured[int(i)].keys()}')
    return logits,captured


def _reference_cache(logits,input_ids,attention_mask):
    valid=attention_mask[:,1:].bool().cpu()
    targets=input_ids[:,1:].cpu()[valid]
    z=logits[:,:-1,:][valid]
    logp=F.log_softmax(z,dim=-1)
    return {
        'valid_mask':valid,'targets':targets,'logits':z,'logp':logp,'prob':logp.exp(),
        'nll':float((-logp.gather(1,targets[:,None]).squeeze(1)).mean()),
        'top1':z.argmax(dim=-1),'rms':float(torch.sqrt(torch.mean(z*z))),
        'token_count':int(targets.numel()),
    }


def _compare_logits(ref,logits):
    z=logits[:,:-1,:][ref['valid_mask']]
    logp=F.log_softmax(z,dim=-1)
    nll=float((-logp.gather(1,ref['targets'][:,None]).squeeze(1)).mean())
    kl=float(torch.mean(torch.sum(ref['prob']*(ref['logp']-logp),dim=-1)))
    delta=z-ref['logits']; rms=float(torch.sqrt(torch.mean(delta*delta)))
    return {
        'next_token_nll_delta':nll-ref['nll'],
        'mean_kl_reference_to_perturbed':kl,
        'top1_flip_fraction':float(torch.mean((z.argmax(dim=-1)!=ref['top1']).float())),
        'relative_rms_logit_delta':rms/max(ref['rms'],1e-30),
        'max_abs_logit_delta':float(torch.max(torch.abs(delta))),
    }


def _qk_patch(model,layer_id,kv_head,rank,q_heads,kv_heads,head_dim):
    layer=model.model.layers[int(layer_id)]; attn=layer.self_attn
    q0=attn.q_proj.weight.detach().clone(); k0=attn.k_proj.weight.detach().clone()
    q=q0.float().cpu().numpy(); k=k0.float().cpu().numpy()
    group=q_heads//kv_heads; qids=list(range(kv_head*group,(kv_head+1)*group))
    mats=[q[h*head_dim:(h+1)*head_dim,:].T.copy() for h in qids]
    mats.append(k[kv_head*head_dim:(kv_head+1)*head_dim,:].T.copy())
    approx=normalized_family_rank_approximation(mats,int(rank))
    with torch.no_grad():
        for h,recon in zip(qids,approx.reconstructed[:group]):
            attn.q_proj.weight[h*head_dim:(h+1)*head_dim,:].copy_(torch.from_numpy(recon.T).to(attn.q_proj.weight.dtype))
        attn.k_proj.weight[kv_head*head_dim:(kv_head+1)*head_dim,:].copy_(torch.from_numpy(approx.reconstructed[-1].T).to(attn.k_proj.weight.dtype))
    def restore():
        with torch.no_grad(): attn.q_proj.weight.copy_(q0); attn.k_proj.weight.copy_(k0)
    return {
        'layer':int(layer_id),'kv_head':int(kv_head),'rank':int(rank),
        'q_heads':qids,'normalized_family_residual':approx.normalized_family_residual,
        'per_member_relative_frobenius_change':list(approx.per_member_relative_frobenius_change),
    },restore


def _summary(result):
    rows=result['candidates']
    structural=[x['perturbation']['normalized_family_residual'] for x in rows]
    kl=[x['final']['mean_kl_reference_to_perturbed'] for x in rows]
    final_rel=[x['final']['relative_rms_logit_delta'] for x in rows]
    local_attn=[x['local']['attention_relative_rms_delta'] for x in rows]
    local_dec=[x['local']['decoder_relative_rms_delta'] for x in rows]
    layer_summary={}
    for layer in result['candidate_grid']['layers']:
        xs=[x for x in rows if x['perturbation']['layer']==layer]
        layer_summary[str(layer)]={
            'median_structural_residual':float(np.median([x['perturbation']['normalized_family_residual'] for x in xs])),
            'median_attention_relative_rms_delta':float(np.median([x['local']['attention_relative_rms_delta'] for x in xs])),
            'median_decoder_relative_rms_delta':float(np.median([x['local']['decoder_relative_rms_delta'] for x in xs])),
            'median_final_relative_rms_logit_delta':float(np.median([x['final']['relative_rms_logit_delta'] for x in xs])),
            'median_mean_kl':float(np.median([x['final']['mean_kl_reference_to_perturbed'] for x in xs])),
            'median_continuation_ratio':float(np.median([x['derived']['continuation_ratio_final_logit_over_decoder'] for x in xs])),
        }
    by_kl=sorted(rows,key=lambda x:x['final']['mean_kl_reference_to_perturbed'])
    by_amp=sorted(rows,key=lambda x:x['derived']['continuation_ratio_final_logit_over_decoder'],reverse=True)
    return {
        'experiment_id':result['experiment_id'],'status':'PASS','evidence_scope':result['decision_scope'],
        'candidate_count':len(rows),'calibration':result['calibration'],
        'correlations':{
            'structural_residual_vs_mean_kl':pearson_correlation(structural,kl),
            'structural_residual_vs_final_relative_rms_logit_delta':pearson_correlation(structural,final_rel),
            'local_attention_relative_rms_vs_final_relative_rms':pearson_correlation(local_attn,final_rel),
            'local_decoder_relative_rms_vs_final_relative_rms':pearson_correlation(local_dec,final_rel),
        },
        'layer_medians':layer_summary,
        'lowest_final_kl_candidates':[
            {'id':x['id'],'structural_residual':x['perturbation']['normalized_family_residual'],'mean_kl':x['final']['mean_kl_reference_to_perturbed'],'continuation_ratio':x['derived']['continuation_ratio_final_logit_over_decoder']} for x in by_kl[:6]
        ],
        'highest_continuation_ratio_candidates':[
            {'id':x['id'],'decoder_relative_rms_delta':x['local']['decoder_relative_rms_delta'],'final_relative_rms_logit_delta':x['final']['relative_rms_logit_delta'],'continuation_ratio':x['derived']['continuation_ratio_final_logit_over_decoder']} for x in by_amp[:6]
        ],
        'forbidden_inference':result['forbidden_inference'],
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-dir',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--contract',default='experiments/transformer/T001_smollm2_135m/continuation_amplification_contract.json')
    args=ap.parse_args(); contract=json.loads(Path(args.contract).read_text()); model_dir=Path(args.model_dir)
    checkpoint=model_dir/'model.safetensors'; actual_sha=sha256_file(checkpoint)
    if actual_sha!=contract['checkpoint_sha256']: raise SystemExit(f'checkpoint SHA mismatch: {actual_sha}')
    cal_contract=json.loads(Path(contract['calibration_source']).read_text()); calibration=cal_contract['calibration']
    torch.manual_seed(int(contract['runtime']['seed']))
    tokenizer=AutoTokenizer.from_pretrained(contract['model_id'],revision=contract['hub_revision'])
    input_ids,attention_mask=_tokenize(tokenizer,calibration)
    token_hash=sha256_json({'input_ids_hash':_tensor_hash(input_ids),'attention_mask_hash':_tensor_hash(attention_mask),'shape':list(input_ids.shape)})
    if token_hash!=contract['expected_tokenized_inputs_sha256']:
        raise SystemExit(f'tokenized calibration drift: {token_hash}')
    model=AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True); model.eval()
    layers=[int(x) for x in contract['candidate_grid']['layers']]
    ref_logits,ref_capture=_forward_capture(model,input_ids,attention_mask,layers); ref=_reference_cache(ref_logits,input_ids,attention_mask)
    valid_local=attention_mask.bool().cpu().numpy(); g=contract['geometry']; rows=[]
    for layer_id in layers:
        for kv in contract['candidate_grid']['kv_heads']:
            perturb,restore=_qk_patch(model,layer_id,int(kv),int(contract['candidate_grid']['rank']),int(g['q_heads']),int(g['kv_heads']),int(g['head_dim']))
            try: logits,capture=_forward_capture(model,input_ids,attention_mask,[layer_id])
            finally: restore()
            attn_rel=relative_rms_delta(ref_capture[layer_id]['attention'].numpy(),capture[layer_id]['attention'].numpy(),valid_local)
            dec_rel=relative_rms_delta(ref_capture[layer_id]['decoder'].numpy(),capture[layer_id]['decoder'].numpy(),valid_local)
            final=_compare_logits(ref,logits); final_rel=final['relative_rms_logit_delta']
            rows.append({
                'id':f'L{layer_id}_qk_kv{int(kv)}_rank{int(contract["candidate_grid"]["rank"])}',
                'perturbation':perturb,
                'local':{'attention_relative_rms_delta':attn_rel,'decoder_relative_rms_delta':dec_rel},
                'final':final,
                'derived':{
                    'within_layer_ratio_decoder_over_attention':safe_ratio(dec_rel,attn_rel),
                    'continuation_ratio_final_logit_over_decoder':safe_ratio(final_rel,dec_rel),
                },
            })
    result={
        'experiment_id':contract['experiment_id'],'status':'PASS','checkpoint_sha256':actual_sha,
        'contract_sha256':sha256_json(contract),'decision_scope':contract['decision_scope'],'candidate_grid':contract['candidate_grid'],
        'calibration':{'valid_next_token_count':ref['token_count'],'tokenized_inputs_sha256':token_hash,'reference_next_token_nll':ref['nll']},
        'candidates':rows,'forbidden_inference':contract['forbidden_inference'],
    }
    summary=_summary(result); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'continuation_amplification_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'continuation_amplification_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True)); return 0


if __name__=='__main__': raise SystemExit(main())
