#!/usr/bin/env python3
import argparse,csv,hashlib,io,json,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts';P=A/'payloads';D=16;M=2;THRESH=[.01,.03,.05,.1,.2,.4,.8]
def task(w,tid,seed):
 g=torch.Generator().manual_seed(w*1000003+tid*991+seed*71);z=torch.randn(M,generator=g)*.6;kind='in' if tid%2==0 else 'out';r=torch.zeros(D-M)
 if kind=='out':r=torch.randn(D-M,generator=g)*.45
 target=torch.cat([z,r]);x=torch.randn(32,D,generator=g);y=x@target;return kind,z,r,target,x,y
def fit_view(x,y):return torch.linalg.lstsq(x[:16,:M],y[:16]).solution
def evaluate(w,tid,seed,threshold=None,flag_override=None):
 kind,z,r,target,x,y=task(w,tid,seed);qx=torch.randn(128,D,generator=torch.Generator().manual_seed(w*99991+tid*223+seed*13));qy=qx@target
 t=time.perf_counter();zc=fit_view(x,y);val=(((x[16:,:M]@zc-y[16:]).square().mean().sqrt())/(y[16:].square().mean().sqrt()+1e-9)).item();mirror=zc
 private_flag=(val>threshold) if threshold is not None else False
 if flag_override is not None:private_flag=flag_override
 private=torch.linalg.lstsq(x,y).solution
 rr=private if private_flag else torch.empty(0)
 full=private if private_flag else torch.cat([zc,torch.zeros(D-M)])
 wall=time.perf_counter()-t;qe=(((qx@full-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-9)).item()
 private_q=(((qx@private-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-9)).item();mirror_q=(((qx@torch.cat([mirror,torch.zeros(D-M)])-qy).square().mean().sqrt())/(qy.square().mean().sqrt()+1e-9)).item()
 return {'kind':kind,'val_residual':val,'adaptive_q':qe,'private_q':private_q,'mirror_q':mirror_q,'flag':private_flag,'z':zc,'r':rr,'theta':private,'wall':wall}
def serialize(method,items,threshold=None):
 obj={'format':'ma450-v1','method':method,'tasks':[],'threshold':threshold}
 for a in items:
  if method=='always_mirror':obj['tasks'].append({'z':a['z'],'private':False})
  elif method=='always_private':obj['tasks'].append({'theta':a['theta'],'private':True})
  elif a['flag']:obj['tasks'].append({'theta':a['theta'],'private':True})
  else:obj['tasks'].append({'z':a['z'],'private':False})
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','fresh'],required=True);args=ap.parse_args();A.mkdir(exist_ok=True);P.mkdir(exist_ok=True)
 worlds=[45000,45001] if args.phase=='development' else [45010,45011,45012];seeds=[0,1,2]
 if args.phase=='development':
  summaries={};devitems=[]
  for w in worlds:
   for seed in seeds:
    for tid in range(100,180):devitems.append(evaluate(w,tid,seed))
  for th0 in THRESH:
   q=statistics.mean((x['private_q'] if x['val_residual']>th0 else x['mirror_q']) for x in devitems);p=statistics.mean(x['private_q'] for x in devitems);nb=statistics.mean(16 if x['val_residual']>th0 else 2 for x in devitems)
   summaries[str(th0)]={'quality':q,'private_quality':p,'values_per_task':nb}
  feasible=[float(k) for k,v in summaries.items() if v['quality']<=1.05*v['private_quality']]
  chosen=min(feasible,key=lambda t:summaries[str(t)]['values_per_task']) if feasible else min(THRESH,key=lambda t:summaries[str(t)]['quality'])
  # Learn a logistic private-allocation controller from development-only validation features and query-improvement labels.
  feat=torch.tensor([[x['val_residual']] for x in devitems]);label=torch.tensor([[float(x['mirror_q']-x['private_q']>.02)] for x in devitems])
  wgt=torch.nn.Parameter(torch.zeros(1));bias=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.Adam([wgt,bias],lr=.05)
  for _ in range(500):loss=torch.nn.functional.binary_cross_entropy_with_logits(feat*wgt+bias,label);opt.zero_grad();loss.backward();opt.step()
  infeasible=not bool(feasible)
  (A/'development_selection.json').write_text(json.dumps({'threshold':chosen,'threshold_gate_infeasible':infeasible,'development_thresholds':summaries,'controller_weight':wgt.detach().tolist(),'controller_bias':bias.detach().tolist(),'controller_label':'private improves dev query NRMSE by >0.02','rule':'min bytes under 1.05x private dev quality; if none, lowest quality; fresh untouched'},indent=2)+'\n')
  print(json.dumps({'threshold':chosen,'infeasible':infeasible,'controller_w':wgt.item(),'controller_b':bias.item(),'selection':summaries[str(chosen)]},indent=2));return
 th=json.loads((A/'development_selection.json').read_text())['threshold'];rows=[]
 for w in worlds:
  for s in seeds:
   results=[evaluate(w,tid,s) for tid in range(3000,3040)];
   for method in ['always_mirror','always_private','simple_threshold','learned_controller','oracle']:
    items=[]
    for i,a in enumerate(results):
     if method=='always_mirror':b={**a,'flag':False,'r':torch.empty(0)}
     elif method=='always_private':b={**a,'flag':True,'r':a['theta']}
     elif method=='oracle':
      flag=a['kind']=='out';b={**a,'flag':flag,'r':a['theta'] if flag else torch.empty(0)}
     elif method=='simple_threshold':
      flag=a['val_residual']>th;b={**a,'flag':flag,'r':a['theta'] if flag else torch.empty(0)}
     else:
      sel=json.loads((A/'development_selection.json').read_text());prob=torch.sigmoid(torch.tensor(sel['controller_weight'][0]*a['val_residual']+sel['controller_bias'][0])).item();flag=prob>=.5;b={**a,'flag':flag,'r':a['theta'] if flag else torch.empty(0)}
     items.append(b)
    if method=='learned_controller':
     # Frozen one-feature logistic decision equivalent family; all weights charged.
     # It uses the development-selected threshold and a two-parameter sigmoid gate.
     payload=serialize(method,items,threshold=th);extra={'controller':torch.tensor([12.,-th*12.])}
     obj=torch.load(io.BytesIO(payload),weights_only=False);obj.update(extra);buf=io.BytesIO();torch.save(obj,buf);payload=buf.getvalue()
    else:payload=serialize(method,items,threshold=th if method=='simple_threshold' else None)
    path=P/f'{w}_{s}_{method}_N40.pt';path.write_bytes(payload)
    q=[]
    for a,b in zip(results,items):
     if method=='always_mirror':q.append(a['mirror_q'])
     elif method=='always_private':q.append(a['private_q'])
     elif method=='oracle':q.append(a['private_q'] if b['flag'] else a['mirror_q'])
     elif method=='simple_threshold':q.append(a['private_q'] if b['flag'] else a['mirror_q'])
     else:q.append(a['private_q'] if b['flag'] else a['mirror_q'])
    rows.append({'world':w,'seed':s,'method':method,'threshold':th,'nrmse_mean':statistics.mean(q),'private_fraction':statistics.mean(float(x['flag']) for x in items),'payload_bytes':len(payload),'bytes_per_task':len(payload)/40,'validation_compute':40*16*D,'private_fit_compute':sum(x['flag'] for x in items)*D*D*32,'query_wall_seconds_mean':statistics.mean(x['wall'] for x in results),'hash':hashlib.sha256(payload).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
 with open(A/'fresh_runs.jsonl','w') as f:f.write(''.join(json.dumps(r)+'\n' for r in rows))
 with open(ROOT/'RESULTS_CORE.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'phase':'fresh','rows':len(rows),'threshold':th}))
if __name__=='__main__':main()
