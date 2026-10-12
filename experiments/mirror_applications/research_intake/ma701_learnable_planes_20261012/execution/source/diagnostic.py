"""Prospectively locked LR diagnostic after primary development failure."""
import argparse,json
from pathlib import Path
import torch
from .runner import read_rows,run_one
from .experiment import METHODS

def diagnostic_grid(method):
    sigmas=[.03] if method in ('shared','multihead','private_ffn') else [.03,.12]
    return [dict(lr=lr,axis_lr=lr,warmup=0,sigma=s,steps=800) for lr in [.001,.003,.01] for s in sigmas]

def calibration():
    root=Path('runs_adjust');selected={};reuse=[]
    for fam in ['aligned','digits8']:
        old=read_rows(Path('runs/calibration')/fam)
        for i,cfg in enumerate(diagnostic_grid('learned')):
            found=any(r['label']=='learned' and r['cfg']==cfg for r in old)
            if not found:run_one(root,'calibration',fam,17100,'learned',cfg,i,800)
        pool=old+read_rows(root/'calibration'/fam);s={}
        for method in METHODS:
            candidates=[r for r in pool if r['label']==method and r['cfg'] in diagnostic_grid(method)]
            assert len(candidates)==len(diagnostic_grid(method)),(fam,method,len(candidates))
            row=min(candidates,key=lambda r:r['val']['loss']);s[method]=row['cfg']
            reuse.append(dict(family=fam,method=method,cfg=row['cfg'],validation_loss=row['val']['loss'],payload_sha=row['payload']['sha256']))
        eligible=[r for r in pool if r['label'] in ['film','lora2','householder','multihead'] and r['cfg'] in diagnostic_grid(r['label'])]
        s['_long_native']=min(eligible,key=lambda r:r['val']['loss'])['label'];selected[fam]=s
    (root/'SELECTED.json').write_text(json.dumps(selected,indent=2,sort_keys=True))
    (root/'SELECTION_PROVENANCE.json').write_text(json.dumps(reuse,indent=2))
    print(json.dumps(selected,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibration','dev']);ap.add_argument('--seed',type=int)
    a=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.stage=='calibration':calibration();return
    if a.seed not in [27101,27102,27103]:raise ValueError('diagnostic seed firewall')
    selected=json.loads(Path('runs_adjust/SELECTED.json').read_text())
    for fam in ['aligned','digits8']:
        s=selected[fam]
        for method in ['learned','fixed_matched','fixed','lora2','householder','multihead','film']:
            cfg=s['learned' if method=='fixed_matched' else method]
            long=method in ['learned','fixed_matched',s['_long_native']]
            run_one(Path('runs_adjust'),'dev',fam,a.seed,method,cfg,0,2400 if long else 800)
if __name__=='__main__':main()
