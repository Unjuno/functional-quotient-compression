"""Independent NumPy inference audit on all locked modular-grokking payloads."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, logsumexp
from model import Config,MirrorSwiGLU,checkpoint_bytes
from train import split_data,source_sha


def numpy_forward(x, state, method, cfg):
    st={k:v.detach().cpu().numpy().astype(np.float64) for k,v in state.items()}
    x=np.array(x,dtype=np.float64)
    p=cfg['p'];h=cfg['h'];q=cfg['q']
    if method=='independent':
        g=np.einsum('nd,kdh->nkh',x,st['wg'])+st['bg'][None,:,:]
        u=np.einsum('nd,kdh->nkh',x,st['wu'])+st['bu'][None,:,:]
        return np.einsum('nkh,khp->nkp',g*expit(g)*u,st['wd'])+st['bo'][None,:,:]
    g=x@st['wg'][0]+st['bg'][0];u=x@st['wu'][0]+st['bu'][0]
    if method=='tied':
        y=(g*expit(g)*u)@st['wd'][0]+st['bo'][0]
        return np.broadcast_to(y[:,None,:],(len(x),4,p)).copy()
    def transform(z,angles,mode):
        qq=angles.shape[1]
        pair=z[:,:2*qq].reshape(len(z),1,qq,2)
        if mode=='rotation':
            a=angles[None,:,:]
            co=np.cos(a);si=np.sin(a)
            first=co*pair[...,0]-si*pair[...,1]
            second=si*pair[...,0]+co*pair[...,1]
            v=np.stack([first,second],axis=-1).reshape(len(z),len(angles),2*qq)
        else:
            v=(pair*(1+angles[None,:,:,None])).reshape(len(z),len(angles),2*qq)
        return np.concatenate([v,np.broadcast_to(z[:,None,2*qq:],(len(z),len(angles),h-2*qq))],axis=-1)
    if method=='pair_mirror':
        gv=transform(g,st['ang_g'],'rotation');uv=transform(u,st['ang_u'],'rotation')
        return (gv*expit(gv)*uv)@st['wd'][0]+st['bo'][0]
    if method in ('gain','mirror'):
        kind='rotation' if method=='mirror' else 'gain'
        gv=transform(g,st['ang_g'],kind);uv=transform(u,st['ang_u'],kind)
        z=gv[:,:,None,:]*expit(gv[:,:,None,:])*uv[:,None,:,:]
        return (z@st['wd'][0]).reshape(len(x),4,p)+st['bo'][0]
    raise ValueError(method)

def evaluate(logits,labels,p):
    ref=logsumexp(logits,axis=-1)-np.take_along_axis(logits,labels[...,None],axis=-1)[...,0]
    return {'ce':float(ref.mean()),'accuracy':float((logits.argmax(-1)==labels).mean()),'min_pair_accuracy':float((logits.argmax(-1)==labels).mean(0).min())}

def audit(root:Path):
    setup=json.loads((root/'setup.json').read_text())
    hashes=setup['training_hashes']
    assert source_sha()==hashes
    outcome=[]
    for file in sorted(root.glob('result_*.json')):
        row=json.loads(file.read_text())
        label=row['label'];w=row['world'];seed=row['seed'];method=row['method']
        model_file=root/f'model_{w}_{seed}_{label}.pt';hist_file=root/f'history_{w}_{seed}_{label}.json'
        raw=model_file.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['payload_sha256']
        saved=torch.load(model_file,weights_only=True,map_location='cpu')
        assert saved['world']==w and saved['seed']==seed and saved['method']==method
        cfg=Config(**saved['cfg'])
        model=MirrorSwiGLU(cfg,method);model.load_state_dict(saved['model']);model.eval()
        assert model.param_count()==row['parameters']
        assert checkpoint_bytes(model)==row['serialized_inference_bytes']
        train,test,_,_=split_data(cfg.p,w,setup['train_fraction'])
        hist=json.loads(hist_file.read_text());assert hist[-1]['step']==setup['steps']
        discrepancies=[]
        for x,y,tag in ((train[0],train[1],'train'),(test[0],test[1],'test')):
            with torch.no_grad(): original=model(x).cpu().numpy()
            independent=numpy_forward(x,saved['model'],method,saved['cfg'])
            discrepancy=float(abs(original-independent).max())
            assert discrepancy<2e-3,(file,tag,discrepancy)
            discrepancies.append(discrepancy)
            m=evaluate(independent,y.numpy(),cfg.p)
            assert abs(m['ce']-row['final_'+tag+'_ce'])<2e-4,(file,tag,m['ce'],row['final_'+tag+'_ce'])
            assert abs(m['accuracy']-row['final_'+tag+'_accuracy'])<1e-6
        outcome.append({'world':w,'seed':seed,'label':label,'num_steps':len(hist)-1,'size':len(raw),
                        'inference_bytes':row['serialized_inference_bytes'],'max_logit_diff':max(discrepancies),
                        'test_acc':row['final_test_accuracy'],'train_acc':row['final_train_accuracy']})
    data={'checked':len(outcome),'all_sha_verified':True,'all_independent_numpy_metrics_verified':True,
          'largest_abs_logit_difference':max(v['max_logit_diff'] for v in outcome),
          'models':outcome}
    (root/'AUDIT.json').write_text(json.dumps(data,indent=2)+'\n')
    print('AUDIT:',len(outcome),'stored inference models independently decoded and checked, max logits',data['largest_abs_logit_difference'])
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--path',required=True)
    audit(Path(parser.parse_args().path))
