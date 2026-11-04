import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D,MODES,RANK,STREAMS,LENGTH=32,4,4,16,128
LR,NOISE=.3,.03
SCHEMA=np.frombuffer(b"TTT-PRIVATE-RESIDUAL-V1",dtype=np.uint8)

def make_world(seed,private_rank):
 rng=np.random.default_rng(seed*100+private_rank)
 base=rng.normal(0,.01,(D,D)).astype(np.float32)
 q,_=np.linalg.qr(rng.normal(size=(D,D)));dictionary=q[:,:MODES*RANK].astype(np.float32);complement=q[:,MODES*RANK:].astype(np.float32)
 streams=[]
 for sid in range(STREAMS):
  mode=sid%MODES;u=dictionary[:,mode*RANK:(mode+1)*RANK]
  if private_rank:
   rot,_=np.linalg.qr(rng.normal(size=(D-MODES*RANK,D-MODES*RANK)))
   v=complement@rot[:,:private_rank]
   cpriv=rng.normal(0,.15,(D,private_rank))
  else:v=np.zeros((D,0));cpriv=np.zeros((D,0))
  cshared=rng.normal(0,.2,(D,RANK));target=base.astype(np.float64)+cshared@u.T+(cpriv@v.T if private_rank else 0)
  x=rng.normal(0,1/np.sqrt(D),(LENGTH,D));y=x@target+rng.normal(0,NOISE,(LENGTH,D))
  streams.append({'mode':mode,'u':u,'v':v,'target':target,'x':x,'y':y})
 return base,dictionary,streams

def choose_mode(base,dictionary,x,y):
 g=(2.0/len(x))*x.T@(x@base-y)
 scores=[np.linalg.norm(g@dictionary[:,m*RANK:(m+1)*RANK]) for m in range(MODES)]
 return int(np.argmax(scores)),int(2*len(x)*D*D+2*MODES*RANK*D)

def online_full(base,s):
 w=base.astype(np.float64).copy();loss=[];mse=[];tail=[]
 for t,(x,y) in enumerate(zip(s['x'],s['y'])):
  e=x@w-y;v=float(np.mean(e*e));mse.append(v)
  if t>=LENGTH-32:tail.append(v)
  loss.append(.5*(D*np.log(2*np.pi*NOISE**2)+v/NOISE**2));w-=LR*2*np.outer(x,e)
 return w-base,np.array([np.mean(loss),np.mean(mse),np.mean(tail)])

def online_basis(base,s,basis):
 b=np.zeros((D,basis.shape[1]),dtype=np.float64);loss=[];mse=[];tail=[]
 for t,(x,y) in enumerate(zip(s['x'],s['y'])):
  e=x@(base+b@basis.T)-y;v=float(np.mean(e*e));mse.append(v)
  if t>=LENGTH-32:tail.append(v)
  loss.append(.5*(D*np.log(2*np.pi*NOISE**2)+v/NOISE**2));b-=LR*2*np.outer(x,e@basis)
 return b,np.array([np.mean(loss),np.mean(mse),np.mean(tail)])

def save(seed,r,name,payload):
 p=ROOT/'source'/f'{seed}_r{r}_{name}.npz';t=time.perf_counter();np.savez_compressed(p,**payload);return p.stat().st_size,time.perf_counter()-t

def run_condition(seed,r):
 base,dictionary,streams=make_world(seed,r);rows=[];packed=[];times={}
 # Full TTT-Linear.
 t=time.perf_counter();deltas=[];metrics=[]
 for s in streams:d,m=online_full(base,s);deltas.append(d.astype(np.float32));metrics.append(m)
 times['full_ttt_linear']=time.perf_counter()-t
 packed.append(('full_ttt_linear',np.mean(metrics,axis=0),np.max(metrics,axis=0),{'schema':SCHEMA,'base_initialization':base,'stream_ids':np.arange(STREAMS,dtype=np.uint8),'writable_deltas':np.stack(deltas)},STREAMS*LENGTH*4*D*D,0))
 # Shared modes only.
 codes=[];shares=[];ml=[];router_ops=0;t=time.perf_counter()
 for s in streams:
  c,op=choose_mode(base,dictionary,s['x'][:8],s['y'][:8]);router_ops+=op;codes.append(c);u=dictionary[:,c*RANK:(c+1)*RANK];b,m=online_basis(base,s,u);shares.append(b.astype(np.float32));ml.append(m)
 times['shared_only']=time.perf_counter()-t
 packed.append(('shared_only',np.mean(ml,axis=0),np.max(ml,axis=0),{'schema':SCHEMA,'base_initialization':base,'shared_dictionary':dictionary,'stream_codes':np.asarray(codes,dtype=np.uint8),'writable_shared_codes':np.stack(shares)},STREAMS*LENGTH*6*D*RANK,router_ops))
 # Mirror and native shared + private residual.
 for name in ('mirror_shared_private','native_dictionary_private'):
  t=time.perf_counter();coeffs=[];privbases=[];losses=[];route=0;localcodes=[]
  for s in streams:
   c,op=choose_mode(base,dictionary,s['x'][:8],s['y'][:8]);route+=op;localcodes.append(c);u=dictionary[:,c*RANK:(c+1)*RANK]
   basis=np.concatenate([u,s['v']],axis=1);b,m=online_basis(base,s,basis);coeffs.append(b.astype(np.float32));privbases.append(s['v'].astype(np.float32));losses.append(m)
  times[name]=time.perf_counter()-t
  payload={'schema':SCHEMA,'base_initialization':base,'shared_dictionary':dictionary,'stream_codes':np.asarray(localcodes,dtype=np.uint8),'private_bases':np.stack(privbases),'writable_shared_private_codes':np.stack(coeffs)}
  if name=='native_dictionary_private':
   other=packed[-1] if False else None
  packed.append((name,np.mean(losses,axis=0),np.max(losses,axis=0),payload,STREAMS*LENGTH*6*D*(RANK+r),route))
 # Independent private basis upper control.
 t=time.perf_counter();pbas=[];pcoeff=[];losses=[]
 for s in streams:
  basis=np.concatenate([s['u'],s['v']],axis=1);b,m=online_basis(base,s,basis);pbas.append(basis.astype(np.float32));pcoeff.append(b.astype(np.float32));losses.append(m)
 times['independent_private']=time.perf_counter()-t
 packed.append(('independent_private',np.mean(losses,axis=0),np.max(losses,axis=0),{'schema':SCHEMA,'base_initialization':base,'stream_ids':np.arange(STREAMS,dtype=np.uint8),'private_bases':np.stack(pbas),'writable_codes':np.stack(pcoeff)},STREAMS*LENGTH*6*D*(RANK+r),0))
 # Native dictionary payload is exactly the shared+private arrays.
 mi=next(x for x in packed if x[0]=='mirror_shared_private'); ni=next(x for x in packed if x[0]=='native_dictionary_private')
 for k,v in mi[3].items():
  if isinstance(v,np.ndarray): assert np.array_equal(v,ni[3][k])
 for name,means,worsts,payload,mac,route in packed:
  size,wall=save(seed,r,name,payload)
  rows.append({'seed':seed,'private_residual_rank':r,'method':name,'serialized_state_bytes':size,'mean_prequential_nll':float(means[0]),'mean_token_mse':float(means[1]),'mean_final32_mse':float(means[2]),'worst_stream_nll':float(worsts[0]),'worst_stream_final32_mse':float(worsts[2]),'stream_count':STREAMS,'sequence_tokens':STREAMS*LENGTH,'optimizer_updates':STREAMS*LENGTH,'update_mac_proxy':int(mac),'router_mac_proxy':int(route),'wall_time_s':times[name],'serialization_wall_s':wall})
 return rows

def run(seeds):
 rows=[r for seed in seeds for rank in (0,2,4,8) for r in run_condition(seed,rank)]
 (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow([f"private_rank_{r['private_residual_rank']}",r['seed'],r['method'],r['serialized_state_bytes'],r['sequence_tokens'],r['optimizer_updates'],r['update_mac_proxy']+r['router_mac_proxy'],r['wall_time_s'],'mean_final32_mse',r['mean_final32_mse'],'mean_prequential_nll',r['mean_prequential_nll'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
