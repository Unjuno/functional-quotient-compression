from __future__ import annotations
import json,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from engine import forward

def eval_passages(state,cfg,reference,probes,device='cpu'):
    rows=[];t=time.perf_counter()
    with torch.inference_mode():
        for p in probes:
            ids=torch.tensor([p['input_ids']],device=device)
            ref=reference[p['id']].to(device);z=forward(state,cfg,ids)[0,:-1]
            lr=F.log_softmax(ref,dim=-1);lc=F.log_softmax(z,dim=-1)
            kl=(lr.exp()*(lr-lc)).sum(-1)
            nll=F.nll_loss(lc,ids[0,1:],reduction='none');refnll=F.nll_loss(lr,ids[0,1:],reduction='none')
            flip=(ref.argmax(-1)!=z.argmax(-1)).float()
            row={'passage_id':p['id'],'split':p['split'],'n_tokens':len(p['input_ids'])-1,'KL':kl.mean().item(),
                 'NLL':nll.mean().item(),'reference_NLL':refnll.mean().item(),'delta_NLL':(nll-refnll).mean().item(),
                 'top1_flip':flip.mean().item(),'finite':bool(torch.isfinite(z).all() and torch.isfinite(kl).all())}
            rows.append(row)
    return {'passages':rows,'elapsed_seconds':time.perf_counter()-t,'summary':summarize(rows)}

def make_reference(state,cfg,probes,device='cpu'):
    refs={}
    with torch.inference_mode():
        for p in probes:
            ids=torch.tensor([p['input_ids']],device=device)
            refs[p['id']]=forward(state,cfg,ids)[0,:-1].cpu().contiguous()
    return refs

def summarize(rows):
    out={'n_passages':len(rows),'n_tokens':sum(r['n_tokens'] for r in rows),'all_finite':all(r['finite'] for r in rows)}
    rng=np.random.default_rng(266272);inds=rng.integers(0,len(rows),size=(4000,len(rows)))
    for key in ['KL','NLL','reference_NLL','delta_NLL','top1_flip']:
        a=np.array([r[key] for r in rows]);means=a[inds].mean(1)
        out[key]={'mean':float(a.mean()),'median':float(np.median(a)),'min':float(a.min()),'max':float(a.max()),
                  'passage_bootstrap_95CI':np.quantile(means,[.025,.975]).tolist(),
                  'standard_error_across_passages':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None}
    return out
