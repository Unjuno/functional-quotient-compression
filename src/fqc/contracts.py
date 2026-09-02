"""Real-pilot and Transformer extraction contract validation.

Canonicalized from D56/D57 and strengthened with the D59 decoder-DAG rules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from .decoder_dag import Atom, compile_decoder_dag
from .linear_orientation import expected_attention_projection_shapes

@dataclass(frozen=True)
class PilotValidation:
    errors: tuple[str,...]; warnings: tuple[str,...]
    unique_paid_scalar_count: int; baseline_bits: int; target_bits_64x: int
    total_paid_atom_bits: int; required_paid_atom_bits: int
    @property
    def valid(self): return not self.errors

@dataclass(frozen=True)
class ExtractionValidation:
    errors: tuple[str,...]; warnings: tuple[str,...]; unique_baseline_scalar_count: int
    @property
    def valid(self): return not self.errors

def _prod(shape):
    n=1
    for d in shape: n*=d
    return n

def validate_pilot_contract(c: Mapping[str,Any], m: Mapping[str,Any]) -> PilotValidation:
    e=[]; w=[]
    N=c.get('model',{}).get('unique_paid_scalar_count_N')
    if not isinstance(N,int) or isinstance(N,bool) or N<=0: e.append('N must be a positive integer'); N=0
    baseline=c.get('baseline',{}); target=c.get('target',{})
    if baseline.get('bits_per_scalar')!=16: e.append('bits_per_scalar must be 16')
    if baseline.get('baseline_bits')!=16*N: e.append('baseline_bits must equal 16*N')
    if target.get('compression_factor')!=64: e.append('compression_factor must be 64')
    if target.get('B64_integer_bits')!=N//4: e.append('B64_integer_bits must equal floor(N/4)')
    for k in ('tensor_inventory','operator_manifest','paid_atom_manifest','quality_contract','replay_contract'):
        v=c.get('artifact_hashes',{}).get(k)
        if not isinstance(v,str) or not v.startswith('sha256:'): e.append(f'invalid artifact hash: {k}')
    raw=m.get('atoms',[]); ids=set(); atoms=[]; total=0
    for a in raw if isinstance(raw,list) else []:
        aid=a.get('atom_id'); cls=a.get('class'); bits=a.get('bits'); deps=a.get('dependencies',[])
        if not isinstance(aid,str) or not aid: e.append('invalid atom_id'); continue
        if aid in ids: e.append(f'duplicate atom_id: {aid}'); continue
        ids.add(aid)
        if not isinstance(bits,int) or isinstance(bits,bool) or bits<0: e.append(f'{aid}: invalid bits'); bits=0
        if cls!='PAID' and bits!=0: e.append(f'{aid}: non-PAID bits must be zero')
        if cls=='PAID': total+=bits
        if not isinstance(deps,list) or any(not isinstance(d,str) for d in deps): e.append(f'{aid}: invalid dependencies'); deps=[]
        atoms.append(Atom(aid,str(cls),bits,tuple(deps)))
    outputs=m.get('required_decode_outputs',[])
    if not isinstance(outputs,list): e.append('required_decode_outputs must be a list'); outputs=[]
    compiled=compile_decoder_dag(atoms,outputs)
    e.extend(x for x in compiled.errors if x not in e)
    reachable=set(compiled.reachable_atoms)
    unused=[a.atom_id for a in atoms if a.atom_class=='PAID' and a.atom_id not in reachable]
    if unused: w.append('serialized PAID atoms outside required closure: '+', '.join(sorted(unused)))
    return PilotValidation(tuple(e),tuple(w),N,16*N,N//4,total,compiled.total_paid_bits)

def validate_transformer_extraction(m: Mapping[str,Any]) -> ExtractionValidation:
    e=[]; w=[]
    for k in ('model_identity','tensor_inventory','modules','external_fixed_state','derived_primitives','extraction_policy'):
        if k not in m: e.append(f'missing top-level field: {k}')
    tensors={}; groups={}
    for t in m.get('tensor_inventory',[]) if isinstance(m.get('tensor_inventory',[]),list) else []:
        tid=t.get('tensor_id'); shape=t.get('shape',[])
        if not isinstance(tid,str) or not tid: e.append('invalid tensor_id'); continue
        if tid in tensors: e.append(f'duplicate tensor_id: {tid}'); continue
        tensors[tid]=t
        if not isinstance(shape,list) or not shape or any(not isinstance(d,int) or d<=0 for d in shape): e.append(f'{tid}: invalid shape')
        if t.get('baseline_included'):
            sg=t.get('storage_group')
            if not isinstance(sg,str) or not sg: e.append(f'{tid}: missing storage_group')
            else: groups.setdefault(sg,[]).append(t)
    N=0
    for sg,ts in groups.items():
        counts=[_prod(t['shape']) for t in ts if isinstance(t.get('shape'),list) and t.get('shape')]
        if len(set(counts))!=1: e.append(f'storage_group {sg}: inconsistent scalar counts'); continue
        if len({t.get('dtype') for t in ts})>1: e.append(f'storage_group {sg}: inconsistent dtypes')
        N+=counts[0]
        if len(ts)>1: w.append(f'storage_group {sg} must be proven from actual checkpoint storage identity')
    modules=m.get('modules',{}) if isinstance(m.get('modules',{}),Mapping) else {}
    for a in modules.get('attention',[]) if isinstance(modules.get('attention',[]),list) else []:
        mid=a.get('module_id','<attn>'); qh=a.get('q_heads'); kvh=a.get('kv_heads'); mp=a.get('q_to_kv_map',[])
        if not isinstance(qh,int) or qh<=0 or not isinstance(kvh,int) or kvh<=0: e.append(f'{mid}: invalid head counts'); continue
        if len(mp)!=qh or any(not isinstance(x,int) or x<0 or x>=kvh for x in mp): e.append(f'{mid}: invalid q_to_kv_map')
        projection_ids=a.get('projection_tensor_ids',{})
        for role,tid in projection_ids.items():
            if tid not in tensors: e.append(f'{mid}: missing projection tensor {role}:{tid}')
        identity=m.get('model_identity',{})
        orientation=identity.get('checkpoint_weight_orientation',identity.get('orientation'))
        d_model=a.get('d_model'); hd=a.get('head_dim'); vd=a.get('value_dim',hd)
        if orientation is not None and all(isinstance(x,int) and x>0 for x in (d_model,hd,vd)):
            try:
                expected=expected_attention_projection_shapes(orientation,d_model,qh,kvh,hd,vd)
            except ValueError:
                w.append(f'{mid}: projection shape validation unsupported for orientation {orientation}')
            else:
                for role,shape_expected in expected.items():
                    tid=projection_ids.get(role)
                    if tid in tensors and tensors[tid].get('shape')!=shape_expected:
                        e.append(f'{mid}: {role} shape {tensors[tid].get("shape")} != expected {shape_expected} for {orientation}')
        elif orientation is not None:
            w.append(f'{mid}: projection shape validation skipped because d_model/head_dim/value_dim metadata is incomplete')
        else:
            w.append(f'{mid}: projection shape validation skipped because checkpoint weight orientation is missing')
        red=a.get('exact_reduction_policy')
        if a.get('position_operator')=='rope' and red=='plain_bilinear_family': e.append(f'{mid}: plain bilinear reduction invalid under RoPE')
        if a.get('qk_norm') not in (None,'none') and red!='factor_preserving_required': e.append(f'{mid}: QK norm requires factor-preserving extraction')
        if a.get('position_operator')=='rope':
            rd=a.get('rope',{}).get('rotary_dim'); hd=a.get('head_dim')
            if not isinstance(rd,int) or not isinstance(hd,int) or rd<0 or rd>hd or rd%2: e.append(f'{mid}: invalid rotary_dim')
    for mlp in modules.get('mlp',[]) if isinstance(modules.get('mlp',[]),list) else []:
        mid=mlp.get('module_id','<mlp>')
        tids=mlp.get('tensor_ids',{})
        if not isinstance(tids,Mapping):
            e.append(f'{mid}: tensor_ids must be a mapping')
        else:
            for role,tid in tids.items():
                if tid not in tensors: e.append(f'{mid}: missing MLP tensor {role}:{tid}')
    for norm in modules.get('normalization',[]) if isinstance(modules.get('normalization',[]),list) else []:
        mid=norm.get('module_id','<norm>')
        tids=norm.get('tensor_ids',[])
        if not isinstance(tids,list) or any(not isinstance(tid,str) for tid in tids):
            e.append(f'{mid}: tensor_ids must be a list of tensor ids')
        else:
            for tid in tids:
                if tid not in tensors: e.append(f'{mid}: missing normalization tensor {tid}')
    pids=set()
    for p in m.get('derived_primitives',[]) if isinstance(m.get('derived_primitives',[]),list) else []:
        pid=p.get('primitive_id')
        if not isinstance(pid,str) or not pid: e.append('invalid primitive_id'); continue
        if pid in pids: e.append(f'duplicate primitive_id: {pid}')
        pids.add(pid)
        for tid in p.get('source_tensor_ids',[]):
            if tid not in tensors: e.append(f'{pid}: missing source tensor {tid}')
        if p.get('analysis_only') and p.get('paid_if_serialized') is False: e.append(f'{pid}: analysis_only does not imply free serialization')
    return ExtractionValidation(tuple(e),tuple(w),N)
