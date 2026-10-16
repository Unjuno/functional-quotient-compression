"""Independent NumPy numerical path and frozen-source, byte, metric replay auditor."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from model import generate_world, DualSwiGLU, infer_serialized
ROOT=Path(__file__).resolve().parents[1]


def independent_reference(arr,x):
    obj=torch.load(__import__('io').BytesIO(arr),weights_only=True,map_location='cpu')
    c=obj['config'];s={k:v.cpu().numpy().astype(np.float64) for k,v in obj['state'].items()}
    x=np.asarray(x,dtype=np.float64)
    K,q,h,d=c['K'],c['q'],c['h'],c['d']
    def si(z): return z/(1+np.exp(-np.clip(z,-60,60)))
    def view(z,angles,kind):
        a=angles[:,:q]
        o=np.broadcast_to(z[:,None,:],(len(z),len(a),h)).copy()
        t=o[:,:,:2*q].reshape(len(z),len(a),q,2)
        aa=t[:,:,:,0].copy();bb=t[:,:,:,1].copy()
        if kind=='rotation':
            cc=np.cos(a)[None];ss=np.sin(a)[None]
            t[:,:,:,0]=cc*aa-ss*bb
            t[:,:,:,1]=ss*aa+cc*bb
        else:
            t[:,:,:,0]=aa*(1+a[None]);t[:,:,:,1]=bb*(1+a[None])
        return o
    if c['method']=='independent':
        gg=np.einsum('bd,kdh->bkh',x,s['wg']);uu=np.einsum('bd,kdh->bkh',x,s['wu'])
        return np.einsum('bkh,khd->bkd',si(gg)*uu,s['wd'])
    gg=x@s['wg'];uu=x@s['wu']
    if c['method']=='tied':
        return np.broadcast_to(((si(gg)*uu)@s['wd'])[:,None,:],(len(x),K*K,d)).copy()
    if c['method']=='factor_lora1':
        g=gg[:,None,:]+np.einsum('bd,kd,kh->bkh',x,s['ag'],s['bg'])
        u=uu[:,None,:]+np.einsum('bd,kd,kh->bkh',x,s['au'],s['bu'])
        return (si(g)[:,:,None,:]*u[:,None,:,:]@s['wd']).reshape(len(x),K*K,d)
    kind='rotation' if 'mirror' in c['method'] else 'gain'
    if c['method'].startswith('pair_'):
        g=view(gg,s['gangle'],kind);u=view(uu,s['uangle'],kind)
        return ((si(g)*u)@s['wd']).reshape(len(x),K*K,d)
    g=np.broadcast_to(gg[:,None,:],(len(x),K,h)) if c['method']=='mirror_up' else view(gg,s['gangle'],kind)
    u=np.broadcast_to(uu[:,None,:],(len(x),K,h)) if c['method']=='mirror_gate' else view(uu,s['uangle'],kind)
    return (si(g)[:,:,None,:]*u[:,None,:,:]@s['wd']).reshape(len(x),K*K,d)


def audit(out,expect=480):
    out=Path(out)
    lock=json.loads((ROOT/'PRE_FRESH_LOCK.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in lock['files_sha256'].items()),'PRE-FRESH SOURCE CHANGED'
    files=sorted((out/'runs').glob('*.json'))
    assert len(files)==expect,(len(files),expect)
    values=[]
    max_pred_delta=0;max_loss_delta=0;max_r=torch.zeros(()).item()
    per_family={}
    for i,path in enumerate(files):
        r=json.loads(path.read_text());world=generate_world(int(r['world']),r['family'].replace('_full','').replace('_holdout',''))
        data=(out/r['model_path']).read_bytes()
        assert len(data)==r['serialized_bytes']
        assert hashlib.sha256(data).hexdigest()==r['sha256']
        x,y=world['test']
        pred=infer_serialized(data,x).cpu().numpy()
        ref=independent_reference(data,x.numpy())
        assert np.isfinite(ref).all()
        d=float(np.max(np.abs(pred-ref)))
        max_pred_delta=max(max_pred_delta,d)
        assert d<5e-5,(r['name'],d)
        scale2=float(world['scale']**2)
        loss=float(np.mean((pred-y.numpy())**2)/scale2)
        ld=abs(loss-r['test_nmse']);max_loss_delta=max(max_loss_delta,ld)
        assert ld<4e-7,(r['name'],ld)
        held=float(np.mean((pred[:,[2,3,7]]-y.numpy()[:,[2,3,7]])**2)/scale2)
        # FP32 reduction order varies; use relative tolerance for out-of-class heldout loss >3.
        assert abs(held-r['test_heldout_pairs_nmse']) < 2e-7+1e-6*abs(held), (r['name'],held,r['test_heldout_pairs_nmse'])
        assert r['updates']==2400 and r['examples_presented']==2400*64
        assert len(r['checkpoints'])==4 or r['method']=='auto_mirror'
        vals={k:r[k] for k in ('name','world','seed','family','method','train_nmse','val_nmse','test_nmse','test_trained_pairs_nmse','test_heldout_pairs_nmse','worst_pair_nmse','output_diversity_ratio','active_q','active_parameters','serialized_bytes','elapsed_s','sha256','updates')}
        vals['expansions']=len(r['expansions'])
        values.append(vals)
        if (i+1)%100==0:print('verified',i+1,'/',expect,flush=True)
    import csv
    with (ROOT/'results'/'FRESH_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    report={'verified':len(values),'expected':expect,'source_lock_match':True,'all_sha256_match':True,'max_abs_prediction_delta_independent_np_f64':max_pred_delta,'max_abs_metric_replay_error':max_loss_delta,
            'model_bytes_total':sum(v['serialized_bytes'] for v in values),
            'methods':sorted(set(v['method'] for v in values)),'families':sorted(set(v['family'] for v in values))}
    (ROOT/'results'/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--out',default=str(ROOT/'results/fresh'));pa.add_argument('--expect',type=int,default=480)
    args=pa.parse_args();print(json.dumps(audit(args.out,args.expect),indent=2))
