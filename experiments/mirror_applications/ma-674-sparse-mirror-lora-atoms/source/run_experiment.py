import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=16;E=M=16;N=512;TOPKS=(2,4,8,16)
def softmax(a):
 z=a-a.max(-1,keepdims=True);v=np.exp(z);return v/v.sum(-1,keepdims=True)
def forward_full(x,base,experts,router,top2=False):
 g=softmax(x@router); update=np.zeros_like(x)
 if top2:
  ids=np.argpartition(g,-2,axis=1)[:,-2:];m=np.zeros_like(g);np.put_along_axis(m,ids,1.,axis=1);g*=m;g/=g.sum(1,keepdims=True)
  for n in range(len(x)):
   for e in ids[n]: update[n]+=g[n,e]*(experts[e]@x[n])
 else:
  out=np.einsum('nd,eod->neo',x,experts);update=np.einsum('ne,neo->no',g,out)
 return x@base.T+update
def forward_atoms(x,base,u,v,c,router,k):
 g=softmax(x@router);coeff=g@c
 ix=np.argpartition(np.abs(coeff),-k,axis=1)[:,-k:]
 val=np.take_along_axis(coeff,ix,axis=1);us=u[ix];vs=v[ix]
 z=np.einsum('nd,nkd->nk',x,vs)
 return x@base.T+np.einsum('nk,nkd->nd',val*z,us)
def run(seeds):
 rows=[]
 for seed in seeds:
  q=np.random.default_rng(seed);base=(q.normal(size=(D,D))/D**.5*.15).astype('f');u=(q.normal(size=(M,D))/D**.5).astype('f');v=(q.normal(size=(M,D))/D**.5).astype('f');c=q.normal(size=(E,M)).astype('f')*.3;router=(q.normal(size=(D,E))/D**.5).astype('f');x=q.normal(size=(N,D)).astype('f')
  atoms=np.einsum('mo,mi->moi',u,v);experts=np.einsum('em,moi->eoi',c,atoms);target=forward_full(x,base,experts,router)
  payload={'full_bank':{'base':base,'router':router,'experts':experts},'full_top2':{'base':base,'router':router,'experts':experts,'top_k':np.array([2],dtype='u1')}}
  methods={'full_bank':lambda:forward_full(x,base,experts,router),'full_top2':lambda:forward_full(x,base,experts,router,True)}
  for k in TOPKS:
   payload[f'mirror_top{k}']={'base':base,'router':router,'atom_left':u,'atom_right':v,'expert_codes':c,'top_k':np.array([k],dtype='u1')}
   payload[f'native_top{k}']={'base':base.copy(),'router':router.copy(),'atom_left':u.copy(),'atom_right':v.copy(),'expert_scales':c.copy(),'top_k':np.array([k],dtype='u1')}
   methods[f'mirror_top{k}']=lambda k=k:forward_atoms(x,base,u,v,c,router,k)
   methods[f'native_top{k}']=lambda k=k:forward_atoms(x,base,u,v,c,router,k)
  for name,p in payload.items():
   path=ROOT/'source'/f'{seed}_{name}.npz';t=time.perf_counter();np.savez_compressed(path,**p);serial=time.perf_counter()-t;t=time.perf_counter();y=methods[name]();wall=time.perf_counter()-t
   if name=='full_bank':mac=N*D*D+N*D*E+N*E*D*D
   elif name=='full_top2':mac=N*D*D+N*D*E+N*2*D*D
   else:
    k=int(name.split('top')[-1]);mac=N*D*D+N*D*E+N*E*M+N*k*D*2
   rows.append({'seed':seed,'residual_rms':0.,'method':name,'serialized_bytes':path.stat().st_size,'output_nrmse':float(np.sqrt(np.mean((y-target)**2)/np.mean(target**2))),'examples':N,'optimizer_updates':0,'active_mac_proxy':int(mac),'inference_wall_time_s':wall,'serialization_wall_time_s':serial})
  for k in TOPKS:
   a,b=methods[f'mirror_top{k}'](),methods[f'native_top{k}']();assert np.array_equal(a,b)
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow(['synthetic',r['seed'],r['method'],r['serialized_bytes'],N,0,r['active_mac_proxy'],r['inference_wall_time_s'],'output_nrmse',r['output_nrmse'],'active_mac_proxy',r['active_mac_proxy'],r['serialization_wall_time_s'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(p.parse_args().seeds),indent=2))
