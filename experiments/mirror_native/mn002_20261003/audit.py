# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Hash-lock the complete predeclared suite, then audit new causal sequences."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from model import TinyLM
from task import make_table,dataset,check_splits
from utils import dump,evaluate
ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze(train_dir: Path, protocol_path: Path, manifest_path: Path):
    if manifest_path.exists(): raise FileExistsError('manifest already exists')
    p=json.loads(protocol_path.read_text()); entries=[]
    for seed in p['seeds']:
        for v in p['variants']:
            base=f'{v["name"]}_seed{seed}'
            cp=train_dir/(base+'.pt'); rp=train_dir/(base+'.json')
            if not cp.is_file() or not rp.is_file():
                raise ValueError(f'incomplete predeclared suite: {base}')
            r=json.loads(rp.read_text())
            if r['checkpoint_sha256']!=sha(cp) or r['protocol_sha256']!=sha(protocol_path):
                raise ValueError('training artifact provenance mismatch')
            entries.append({'name':v['name'],'seed':seed,'checkpoint':cp.name,'result':rp.name,
                            'checkpoint_sha256':sha(cp),'result_sha256':sha(rp)})
    dump(manifest_path,{'experiment':p.get('experiment','MN002'),'protocol_sha256':sha(protocol_path),'entries':entries,
                        'meaning':'All predeclared final-step checkpoints fixed before final audit generation; not external attestation.'})
    return entries

def verify_manifest(train_dir: Path, protocol_path: Path, manifest_path: Path):
    m=json.loads(manifest_path.read_text())
    if m['protocol_sha256']!=sha(protocol_path): raise ValueError('protocol changed after freeze')
    p=json.loads(protocol_path.read_text())
    expected={(v['name'],s) for v in p['variants'] for s in p['seeds']}
    if len(m['entries'])!=len(expected) or {(r['name'],r['seed']) for r in m['entries']}!=expected:
        raise ValueError('manifest suite mismatch')
    for r in m['entries']:
        for file_key,hash_key in [('checkpoint','checkpoint_sha256'),('result','result_sha256')]:
            if Path(r[file_key]).name!=r[file_key] or sha(train_dir/r[file_key])!=r[hash_key]:
                raise ValueError('artifact changed after freeze')
    return m['entries']

@torch.no_grad()
def cache_checks(model: TinyLM, probe: torch.Tensor, tolerance: float):
    full=model(probe)[0]; measurements=[]
    for chunk in (1,3,5):
        outputs=[]; past=None
        for i in range(0,probe.shape[1],chunk):
            z,past,_=model(probe[:,i:i+chunk],past=past);outputs.append(z)
        cached=torch.cat(outputs,dim=1)
        error=(full-cached).abs().max().item()
        equal=bool(torch.equal(full.argmax(-1),cached.argmax(-1)))
        if error>tolerance or not equal:
            raise AssertionError({'error':error,'argmax_equal':equal,'chunk':chunk})
        measurements.append({'chunk':chunk,'max_logit_error':error,'all_argmax_equal':equal})
    return {'comparisons':measurements,'kv_bytes':sum(t.numel()*t.element_size() for kv in past for t in kv),
            'sequences':len(probe),'tokens_per_sequence':probe.shape[1]}

def audit_all(train_dir: Path, protocol_path: Path, manifest_path: Path, output: Path):
    entries=verify_manifest(train_dir,protocol_path,manifest_path)
    p=json.loads(protocol_path.read_text())
    torch.set_num_threads(p['threads']);torch.use_deterministic_algorithms(True)
    table=make_table(p['rules'],p['symbols'],p['table_seed'])
    tr=dataset(p['train_sequences'],p['train_seed'],table,p['records'])
    dv=dataset(p['development_sequences'],p['development_seed'],table,p['records'])
    test=dataset(p['audit_sequences'],p['audit_seed'],table,p['records'])
    splits=check_splits(tr,dv,test)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in entries:
        rp=output/(Path(e['result']).stem+'.json')
        if rp.exists(): raise FileExistsError('audit already exists; use a new output and label any reanalysis')
        ck=torch.load(train_dir/e['checkpoint'],weights_only=True,map_location='cpu')
        train=json.loads((train_dir/e['result']).read_text())
        if ck['protocol_sha256']!=sha(protocol_path) or ck['config']!=train['config']:
            raise ValueError('saved model configuration mismatch')
        model=TinyLM(**ck['config']).eval();model.load_state_dict(ck['state_dict'])
        normal=evaluate(model,test)
        controls={}
        if ck['config']['kind'] in ('mirror','full_moe'):
            fixed=[torch.tensor(r,dtype=torch.float32) for r in train['fixed_development_routing']]
            controls['fixed_development_mean']=evaluate(model,test,override=fixed)
            controls['uniform']=evaluate(model,test,override='uniform')
            controls['state_identity_roll']=evaluate(model,test,override='roll')
        cache=cache_checks(model,test[0][:4],5e-5)
        routing=[]
        for r in normal.get('routing',[]):
            mean=torch.tensor(r['mean_prob'],dtype=torch.float64)
            entropy=-(mean*mean.clamp_min(1e-30).log()).sum().item()
            routing.append({'marginal_entropy':entropy,'mean_conditional_entropy':r['mean_entropy'],
                            'input_route_mutual_information':max(0.,entropy-r['mean_entropy']),
                            'argmax_nonempty_states':sum(c>0 for c in r['argmax_counts']),
                            'interpretation':'Diagnostic of induced categorical routing, not independent expert capacity.'})
        result={'experiment':p['experiment']+'_AUDIT','name':e['name'],'seed':e['seed'],'config':ck['config'],
                'checkpoint_sha256':e['checkpoint_sha256'],'manifest_sha256':sha(manifest_path),
                'normal':normal,'ablations':controls,'cache':cache,'routing_diagnostics':routing,
                'data':splits,'parameters':train['parameters'],'parameter_bytes_fp32':train['parameter_bytes_fp32'],
                'checkpoint_bytes':train['checkpoint_bytes'],
                'ff_linear_macs_per_token_all_layers':train['ff_linear_macs_per_token_all_layers']}
        dump(rp,result)
        row={k:result[k] for k in ('name','seed','parameters','parameter_bytes_fp32','checkpoint_bytes','ff_linear_macs_per_token_all_layers')}
        row.update({'audit_nll':normal['nll'],'audit_accuracy':normal['accuracy'],
                    'cache_max_error':max(c['max_logit_error'] for c in cache['comparisons'])})
        if controls:
            row['fixed_routing_nll']=controls['fixed_development_mean']['nll']
            row['fixed_routing_accuracy']=controls['fixed_development_mean']['accuracy']
            row['uniform_accuracy']=controls['uniform']['accuracy']
            row['rolled_accuracy']=controls['state_identity_roll']['accuracy']
        rows.append(row);print(json.dumps(row),flush=True)
    dump(output/'summary.json',rows)
    import csv
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['freeze','run'])
    ap.add_argument('--train-dir',type=Path,default=ROOT/'results/train')
    ap.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    ap.add_argument('--manifest',type=Path,default=ROOT/'results/FROZEN_CHECKPOINTS.json')
    ap.add_argument('--output',type=Path,default=ROOT/'results/audit')
    a=ap.parse_args()
    if a.action=='freeze': freeze(a.train_dir,a.protocol,a.manifest)
    else: audit_all(a.train_dir,a.protocol,a.manifest,a.output)
