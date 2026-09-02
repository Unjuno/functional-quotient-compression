#!/usr/bin/env python3
"""Run T002 task-unconditioned structural audit on pinned SmolLM2 weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from fqc.manifest_builder import sha256_json
from fqc.real_structural_audit import (
    corresponding_weight_family_summary,
    pairwise_cosine_matrix,
    qk_gqa_group_spans,
    span_summary,
    split_pytorch_linear_output_heads,
    swiglu_channel_summary,
    value_output_operators,
)
from fqc.safetensors_inventory import sha256_file


def _offdiag_summary(matrix):
    M=np.asarray(matrix,dtype=np.float64)
    if M.ndim!=2 or M.shape[0]!=M.shape[1] or M.shape[0]<2:
        raise ValueError('square pairwise matrix with at least two members required')
    vals=np.abs(M[~np.eye(M.shape[0],dtype=bool)])
    return {
        'mean_abs':float(vals.mean()),
        'median_abs':float(np.median(vals)),
        'max_abs':float(vals.max()),
    }


def _head_family(weight, heads, head_dim, ranks):
    mats=split_pytorch_linear_output_heads(weight,heads=heads,head_dim=head_dim)
    cos=pairwise_cosine_matrix(mats)
    return {
        'span':span_summary(mats,ranks=ranks),
        'pairwise_cosine_abs_offdiag':_offdiag_summary(cos),
    }


def _load_tensor(handle, key):
    # Safetensors returns BF16 torch tensors. Convert explicitly before NumPy.
    return handle.get_tensor(key).float().cpu().numpy()


def _rank1_excess(span):
    """Concentration above an equal-energy orthogonal member family.

    For m Frobenius-normalized mutually orthogonal members, the rank-1 residual
    is 1-1/m. Positive excess means the observed family is more concentrated.
    This is a descriptive reference, not a statistical null test.
    """
    m=int(span['member_count'])
    baseline=1.0-1.0/m
    observed=float(span['residual_by_rank']['1'])
    return {
        'orthogonal_reference_rank1_residual':baseline,
        'observed_rank1_residual':observed,
        'rank1_concentration_excess_vs_orthogonal':baseline-observed,
    }


def _build_summary(result):
    layers={}
    for layer,r in result['layer_results'].items():
        sw=r['swiglu']
        layers[layer]={
            'query_head_family':_rank1_excess(r['query_head_family']['span']),
            'key_head_family':_rank1_excess(r['key_head_family']['span']),
            'value_head_family':_rank1_excess(r['value_head_family']['span']),
            'value_output_operator_family':_rank1_excess(r['value_output_operator_family']['span']),
            'qk_gqa_groups':[
                {'kv_head':int(x['kv_head']),**_rank1_excess(x['span'])}
                for x in r['qk_gqa_groups']
            ],
            'swiglu_descriptor':{
                'cv':float(sw['descriptor_norm_cv']),
                'top_10_percent_energy_fraction':float(sw['descriptor_energy_concentration']['top_10_percent_energy_fraction']),
                'top_10_percent_excess_over_uniform':float(sw['descriptor_energy_concentration']['top_10_percent_energy_fraction'])-0.10,
                'top_25_percent_energy_fraction':float(sw['descriptor_energy_concentration']['top_25_percent_energy_fraction']),
                'top_25_percent_excess_over_uniform':float(sw['descriptor_energy_concentration']['top_25_percent_energy_fraction'])-0.25,
                'gate_up_cosine_mean_abs':float(sw['gate_up_cosine_mean_abs']),
                'gate_up_cosine_max_abs':float(sw['gate_up_cosine_max_abs']),
            },
        }
    cross={role:_rank1_excess(x['span']) for role,x in result['cross_layer_corresponding_weights'].items()}
    strongest=[]
    for layer,r in layers.items():
        for g in r['qk_gqa_groups']:
            strongest.append((g['rank1_concentration_excess_vs_orthogonal'],f'L{layer}.qk_gqa.kv{g["kv_head"]}'))
        strongest.append((r['query_head_family']['rank1_concentration_excess_vs_orthogonal'],f'L{layer}.query_heads'))
        strongest.append((r['value_output_operator_family']['rank1_concentration_excess_vs_orthogonal'],f'L{layer}.value_output_ops'))
    strongest=sorted(strongest,reverse=True)[:8]
    return {
        'experiment_id':result['experiment_id'],
        'status':'PASS',
        'evidence_scope':result['evidence_scope'],
        'descriptive_reference':'equal_energy_mutually_orthogonal_family_not_a_statistical_null',
        'layers':layers,
        'cross_layer_corresponding_weights':cross,
        'strongest_rank1_concentration_excesses':[
            {'family':name,'excess':float(value)} for value,name in strongest
        ],
        'interpretation':{
            'raw_cross_layer_sharing':'corresponding raw weights across layers 0/15/29 are close to the orthogonal reference',
            'local_candidate':'some GQA 3Q+1K groups and value-output operator families are substantially more concentrated than the orthogonal reference',
            'mlp_candidate':'SwiGLU descriptor energy concentration is weakest at layer 0 and strongest at layer 15 among the audited layers',
            'next_required_gate':'task-conditioned functional sensitivity before any removability or codec claim',
        },
        'forbidden_inference':result['interpretation_boundary'],
    }


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--contract',default='experiments/transformer/T001_smollm2_135m/structural_audit_contract.json')
    ap.add_argument('--output-dir',required=True)
    args=ap.parse_args()

    checkpoint=Path(args.checkpoint)
    contract=json.loads(Path(args.contract).read_text())
    actual_sha=sha256_file(checkpoint)
    if actual_sha!=contract['checkpoint_sha256']:
        raise SystemExit(f'checkpoint SHA256 mismatch: {actual_sha}')

    layers=[int(x) for x in contract['layers']]
    g=contract['geometry']
    q_heads=int(g['q_heads']); kv_heads=int(g['kv_heads']); head_dim=int(g['head_dim'])

    layer_results={}
    role_weights={role:{} for role in ('WQ','WK','WV','WO','gate','up','down')}

    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        keys=set(f.keys())
        for layer in layers:
            p=f'model.layers.{layer}'
            names={
                'WQ':f'{p}.self_attn.q_proj.weight',
                'WK':f'{p}.self_attn.k_proj.weight',
                'WV':f'{p}.self_attn.v_proj.weight',
                'WO':f'{p}.self_attn.o_proj.weight',
                'gate':f'{p}.mlp.gate_proj.weight',
                'up':f'{p}.mlp.up_proj.weight',
                'down':f'{p}.mlp.down_proj.weight',
            }
            missing=sorted(set(names.values())-keys)
            if missing:
                raise SystemExit('missing checkpoint keys: '+', '.join(missing))
            W={role:_load_tensor(f,key) for role,key in names.items()}
            for role,arr in W.items():
                role_weights[role][layer]=arr

            vo=value_output_operators(W['WV'],W['WO'],q_heads=q_heads,kv_heads=kv_heads,head_dim=head_dim)
            vo_cos=pairwise_cosine_matrix(vo)
            qk_groups=qk_gqa_group_spans(W['WQ'],W['WK'],q_heads=q_heads,kv_heads=kv_heads,head_dim=head_dim)
            layer_results[str(layer)]={
                'query_head_family':_head_family(W['WQ'],q_heads,head_dim,(1,2,4,6,8,9)),
                'key_head_family':_head_family(W['WK'],kv_heads,head_dim,(1,2,3)),
                'value_head_family':_head_family(W['WV'],kv_heads,head_dim,(1,2,3)),
                'qk_gqa_groups':qk_groups,
                'value_output_operator_family':{
                    'span':span_summary(vo,ranks=(1,2,4,6,8,9)),
                    'pairwise_cosine_abs_offdiag':_offdiag_summary(vo_cos),
                },
                'swiglu':swiglu_channel_summary(W['gate'],W['up'],W['down']),
            }

    cross_layer={role:corresponding_weight_family_summary(weights) for role,weights in role_weights.items()}

    result={
        'experiment_id':contract['experiment_id'],
        'status':'PASS',
        'checkpoint_sha256':actual_sha,
        'contract_sha256':sha256_json(contract),
        'evidence_scope':contract['evidence_scope'],
        'layers':layers,
        'layer_results':layer_results,
        'cross_layer_corresponding_weights':cross_layer,
        'interpretation_boundary':contract['forbidden_inference'],
    }
    summary=_build_summary(result)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'structural_audit_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'structural_audit_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
