#!/usr/bin/env python3
"""Run T003 task-conditioned functional interventions on pinned SmolLM2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fqc.functional_interventions import (
    bottom_fraction_indices,
    normalized_family_rank_approximation,
    removed_descriptor_energy_fraction,
    swiglu_descriptor_scores,
)
from fqc.manifest_builder import sha256_json
from fqc.safetensors_inventory import sha256_file


def _tensor_hash(t: torch.Tensor) -> str:
    a=t.detach().cpu().contiguous().numpy()
    h=hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(repr(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return 'sha256:'+h.hexdigest()


def _tokenize(tokenizer, texts, max_length: int):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError('tokenizer has neither pad nor eos token')
        tokenizer.pad_token=tokenizer.eos_token
    batch=tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    return batch['input_ids'],batch['attention_mask']


def _forward_logits(model, input_ids, attention_mask):
    with torch.no_grad():
        return model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).logits.float().cpu()


def _reference_cache(logits, input_ids, attention_mask):
    valid=attention_mask[:,1:].bool().cpu()
    targets=input_ids[:,1:].cpu()[valid]
    z=logits[:,:-1,:][valid]
    logp=F.log_softmax(z,dim=-1)
    nll=float((-logp.gather(1,targets[:,None]).squeeze(1)).mean())
    return {
        'valid_mask':valid,
        'targets':targets,
        'logits':z,
        'logp':logp,
        'prob':logp.exp(),
        'nll':nll,
        'top1':z.argmax(dim=-1),
        'rms':float(torch.sqrt(torch.mean(z*z))),
        'token_count':int(targets.numel()),
    }


def _compare(ref, perturbed_logits):
    z=perturbed_logits[:,:-1,:][ref['valid_mask']]
    logp=F.log_softmax(z,dim=-1)
    nll=float((-logp.gather(1,ref['targets'][:,None]).squeeze(1)).mean())
    kl=float(torch.mean(torch.sum(ref['prob']*(ref['logp']-logp),dim=-1)))
    delta=z-ref['logits']
    rms=float(torch.sqrt(torch.mean(delta*delta)))
    max_abs=float(torch.max(torch.abs(delta)))
    flips=float(torch.mean((z.argmax(dim=-1)!=ref['top1']).float()))
    return {
        'reference_nll':ref['nll'],
        'perturbed_nll':nll,
        'next_token_nll_delta':nll-ref['nll'],
        'mean_kl_reference_to_perturbed':kl,
        'top1_flip_fraction':flips,
        'rms_logit_delta':rms,
        'relative_rms_logit_delta':rms/max(ref['rms'],1e-30),
        'max_abs_logit_delta':max_abs,
    }


def _qk_patch(model, spec, q_heads=9, kv_heads=3, head_dim=64):
    layer=model.model.layers[int(spec['layer'])]
    attn=layer.self_attn
    q0=attn.q_proj.weight.detach().clone(); k0=attn.k_proj.weight.detach().clone()
    q=q0.float().cpu().numpy(); k=k0.float().cpu().numpy()
    kv=int(spec['kv_head']); group=q_heads//kv_heads
    qids=list(range(kv*group,(kv+1)*group))
    mats=[]
    for h in qids:
        mats.append(q[h*head_dim:(h+1)*head_dim,:].T.copy())
    mats.append(k[kv*head_dim:(kv+1)*head_dim,:].T.copy())
    approx=normalized_family_rank_approximation(mats,int(spec['rank']))
    expected=float(spec['expected_normalized_family_residual'])
    if abs(approx.normalized_family_residual-expected)>1e-9:
        raise RuntimeError(f'{spec["id"]}: structural residual drift {approx.normalized_family_residual} != {expected}')
    with torch.no_grad():
        for h,recon in zip(qids,approx.reconstructed[:group]):
            attn.q_proj.weight[h*head_dim:(h+1)*head_dim,:].copy_(
                torch.from_numpy(recon.T).to(dtype=attn.q_proj.weight.dtype)
            )
        attn.k_proj.weight[kv*head_dim:(kv+1)*head_dim,:].copy_(
            torch.from_numpy(approx.reconstructed[-1].T).to(dtype=attn.k_proj.weight.dtype)
        )
    d_model=int(q.shape[1])
    info={
        'kind':'qk_family_rank',
        'layer':int(spec['layer']),
        'kv_head':kv,
        'q_heads':qids,
        'family_rank':int(spec['rank']),
        'normalized_family_residual':approx.normalized_family_residual,
        'per_member_relative_frobenius_change':list(approx.per_member_relative_frobenius_change),
        'modified_scalar_count':int((group+1)*d_model*head_dim),
    }
    def restore():
        with torch.no_grad():
            attn.q_proj.weight.copy_(q0); attn.k_proj.weight.copy_(k0)
    return info,restore


def _mlp_patch(model, spec):
    layer=model.model.layers[int(spec['layer'])]
    mlp=layer.mlp
    g0=mlp.gate_proj.weight.detach().clone()
    u0=mlp.up_proj.weight.detach().clone()
    d0=mlp.down_proj.weight.detach().clone()
    U=u0.float().cpu().numpy(); D=d0.float().cpu().numpy(); G=g0.float().cpu().numpy()
    scores=swiglu_descriptor_scores(U,D)
    idx=bottom_fraction_indices(scores,float(spec['fraction']))
    removed_descriptor=removed_descriptor_energy_fraction(scores,idx)
    total_energy=float(np.sum(G*G)+np.sum(U*U)+np.sum(D*D))
    removed_energy=float(np.sum(G[idx]*G[idx])+np.sum(U[idx]*U[idx])+np.sum(D[:,idx]*D[:,idx]))
    tidx=torch.as_tensor(idx,dtype=torch.long)
    with torch.no_grad():
        mlp.gate_proj.weight[tidx,:]=0
        mlp.up_proj.weight[tidx,:]=0
        mlp.down_proj.weight[:,tidx]=0
    d_model=int(U.shape[1])
    info={
        'kind':'swiglu_drop_bottom_descriptor',
        'layer':int(spec['layer']),
        'requested_fraction':float(spec['fraction']),
        'removed_channel_count':int(len(idx)),
        'channel_count':int(len(scores)),
        'actual_channel_fraction':float(len(idx)/len(scores)),
        'removed_descriptor_energy_fraction':removed_descriptor,
        'removed_parameter_frobenius_energy_fraction':removed_energy/max(total_energy,1e-30),
        'modified_scalar_count':int(len(idx)*3*d_model),
    }
    def restore():
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(g0); mlp.up_proj.weight.copy_(u0); mlp.down_proj.weight.copy_(d0)
    return info,restore


def _summary(result):
    rows=[]
    for x in result['interventions']:
        m=x['functional_metrics']; p=x['perturbation']
        row={
            'id':x['id'],
            'kind':p['kind'],
            'next_token_nll_delta':m['next_token_nll_delta'],
            'mean_kl':m['mean_kl_reference_to_perturbed'],
            'top1_flip_fraction':m['top1_flip_fraction'],
            'relative_rms_logit_delta':m['relative_rms_logit_delta'],
        }
        if p['kind']=='qk_family_rank':
            row['structural_residual']=p['normalized_family_residual']
        else:
            row['removed_channel_fraction']=p['actual_channel_fraction']
            row['removed_descriptor_energy_fraction']=p['removed_descriptor_energy_fraction']
            row['removed_parameter_energy_fraction']=p['removed_parameter_frobenius_energy_fraction']
        rows.append(row)
    return {
        'experiment_id':result['experiment_id'],
        'status':result['status'],
        'evidence_scope':result['decision_scope'],
        'calibration_token_count':result['calibration']['valid_next_token_count'],
        'reference_nll':result['reference']['next_token_nll'],
        'interventions':rows,
        'forbidden_inference':result['forbidden_inference'],
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',required=True)
    ap.add_argument('--contract',default='experiments/transformer/T001_smollm2_135m/functional_audit_contract.json')
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()
    contract=json.loads(Path(args.contract).read_text())
    model_dir=Path(args.model_dir)
    checkpoint=model_dir/'model.safetensors'
    actual_sha=sha256_file(checkpoint)
    if actual_sha!=contract['checkpoint_sha256']:
        raise SystemExit(f'checkpoint SHA mismatch: {actual_sha}')

    torch.manual_seed(int(contract['runtime']['seed']))
    tokenizer=AutoTokenizer.from_pretrained(contract['model_id'],revision=contract['hub_revision'])
    input_ids,attention_mask=_tokenize(tokenizer,contract['calibration']['texts'],int(contract['calibration']['max_length']))
    token_hash=sha256_json({
        'input_ids_hash':_tensor_hash(input_ids),
        'attention_mask_hash':_tensor_hash(attention_mask),
        'shape':list(input_ids.shape),
    })

    model=AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        local_files_only=True,
    )
    model.eval()
    ref_logits=_forward_logits(model,input_ids,attention_mask)
    ref=_reference_cache(ref_logits,input_ids,attention_mask)

    outcomes=[]
    for spec in contract['interventions']:
        if spec['kind']=='qk_family_rank':
            perturb,restore=_qk_patch(model,spec)
        elif spec['kind']=='swiglu_drop_bottom_descriptor':
            perturb,restore=_mlp_patch(model,spec)
        else:
            raise RuntimeError(f'unknown intervention kind: {spec["kind"]}')
        try:
            pert_logits=_forward_logits(model,input_ids,attention_mask)
            metrics=_compare(ref,pert_logits)
        finally:
            restore()
        outcomes.append({'id':spec['id'],'perturbation':perturb,'functional_metrics':metrics})

    result={
        'experiment_id':contract['experiment_id'],
        'status':'PASS',
        'checkpoint_sha256':actual_sha,
        'contract_sha256':sha256_json(contract),
        'runtime':contract['runtime'],
        'decision_scope':contract['decision_scope'],
        'calibration':{
            'kind':contract['calibration']['kind'],
            'text_count':len(contract['calibration']['texts']),
            'tokenized_shape':list(input_ids.shape),
            'valid_next_token_count':ref['token_count'],
            'tokenized_inputs_sha256':token_hash,
            'tokenizer_class':type(tokenizer).__name__,
            'tokenizer_vocab_size':int(len(tokenizer)),
        },
        'reference':{
            'next_token_nll':ref['nll'],
            'logit_rms':ref['rms'],
        },
        'interventions':outcomes,
        'forbidden_inference':contract['forbidden_inference'],
    }
    summary=_summary(result)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'functional_audit_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'functional_audit_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
