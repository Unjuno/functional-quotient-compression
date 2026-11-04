import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NKEY,D,RANK,CACHE=64,32,4,8
THRESHOLD,ETA,NOISE=.12,1.0,.01
SCHEMA=np.frombuffer(b"TITANS-LTM-STM-V1",dtype=np.uint8)

def make_world(seed,private_rank):
 rng=np.random.default_rng(seed);q,_=np.linalg.qr(rng.normal(size=(D,D)))
 shared=q[:,:RANK].astype(np.float32);private=q[:,RANK:RANK+private_rank].astype(np.float32)
 base=np.zeros((D,D),dtype=np.float32)
 shared_codes=rng.normal(0,.25,(NKEY,RANK));private_codes=rng.normal(0,.15,(NKEY,8))
 values=[]
 for key in range(NKEY):
  c=shared_codes[key];p=private_codes[key,:private_rank];v=c@shared.T+(p@private.T if private_rank else 0);values.append(v)
 values=np.asarray(values,dtype=np.float32);order=rng.permutation(NKEY);events=[]
 for key in order:
  events.append((int(key),values[key].copy()))
  events.append((int(key),values[key]+rng.normal(0,NOISE,D).astype(np.float32)))
 return base,shared,private,values,events

def surprise_write(matrix,key,value):
 err=value-matrix[key];sur=float(np.linalg.norm(err));count=0
 if sur>THRESHOLD:matrix[key]+=ETA*err;count=1
 return sur,count

def factor_write(codes,private_codes,key,value,shared,private):
 pred=codes[key]@shared.T+(private_codes[key]@private.T if private.shape[1] else 0)
 err=value-pred;sur=float(np.linalg.norm(err));count=0
 if sur>THRESHOLD:
  codes[key]+=ETA*(err@shared)
  if private.shape[1]:private_codes[key]+=ETA*(err@private)
  count=1
 return sur,count

def evaluate(long_values,values,cache_keys,cache_values):
 cache={int(k):cache_values[i] for i,k in enumerate(cache_keys)};lt=[];aware=[]
 for key in range(NKEY):
  lt.append(float(np.mean((long_values[key]-values[key])**2)))
  pred=cache[key] if key in cache else long_values[key]
  aware.append(float(np.mean((pred-values[key])**2)))
 lt=np.asarray(lt);aware=np.asarray(aware)
 nll=lambda m:.5*(D*np.log(2*np.pi*NOISE**2)+D*m/NOISE**2)
 return float(lt.mean()),float(aware.mean()),float(nll(aware.mean()))

def save(seed,r,name,payload,longterm,cache):
 def one(path,obj):
  t=time.perf_counter();np.savez_compressed(path,**obj);return path.stat().st_size,time.perf_counter()-t
 p=ROOT/'source'/f'{seed}_r{r}_{name}_complete.npz';full,wall=one(p,payload)
 lp=ROOT/'source'/f'{seed}_r{r}_{name}_longterm.npz';lb,_=one(lp,longterm)
 cp=ROOT/'source'/f'{seed}_r{r}_{name}_cache.npz';cb,_=one(cp,cache)
 return full,lb,cb,wall

def run_condition(seed,r):
 base,shared,private,values,events=make_world(seed,r);recent=[];seen=set()
 for k,v in reversed(events):
  if k not in seen:
   recent.append((k,v));seen.add(k)
  if len(recent)==CACHE:break
 recent=list(reversed(recent));cache_keys=np.asarray([k for k,_ in recent],dtype=np.uint8);cache_values=np.stack([v for _,v in recent])
 methods=[]
 # Full writable fast-weight matrix.
 t=time.perf_counter();matrix=np.zeros((NKEY,D),dtype=np.float32);writes=0;surprises=[]
 for key,val in events:
  s,c=surprise_write(matrix,key,val);surprises.append(s);writes+=c
 elapsed=time.perf_counter()-t
 common={'schema':SCHEMA,'base_initialization':base,'key_ids':np.arange(NKEY,dtype=np.uint8),'surprise_threshold':np.asarray([THRESHOLD],dtype=np.float32)}
 full_long={**common,'surprise_write_total':np.asarray([writes],dtype=np.uint16),'memory_matrix':matrix};cache={'schema':SCHEMA,'cache_keys':cache_keys,'cache_values':cache_values}
 full_payload={**full_long,**cache}
 methods.append(('titans_full_matrix',matrix,full_payload,full_long,writes,elapsed,NKEY*D*4))
 # Shared-only rank-4 factor.
 t=time.perf_counter();codes=np.zeros((NKEY,RANK),dtype=np.float32);writes=0
 for key,val in events:
  _,c=factor_write(codes,np.zeros((NKEY,0),dtype=np.float32),key,val,shared,np.zeros((D,0)));writes+=c
 elapsed=time.perf_counter()-t;long_values=codes@shared.T
 shared_long={**common,'surprise_write_total':np.asarray([writes],dtype=np.uint16),'shared_value_basis':shared,'key_codes':codes};shared_payload={**shared_long,**cache}
 methods.append(('shared_only_rank4',long_values,shared_payload,shared_long,writes,elapsed,NKEY*RANK*D*4))
 # Shared + private residual, and the exact native factor control.
 for name in ('mirror_shared_private','native_lowrank_memory'):
  t=time.perf_counter();cshared=np.zeros((NKEY,RANK),dtype=np.float32);cpriv=np.zeros((NKEY,r),dtype=np.float32);writes=0
  for key,val in events:
   _,c=factor_write(cshared,cpriv,key,val,shared,private);writes+=c
  elapsed=time.perf_counter()-t;decoded=cshared@shared.T+(cpriv@private.T if r else 0)
  long={**common,'surprise_write_total':np.asarray([writes],dtype=np.uint16),'shared_value_basis':shared,'key_codes':cshared,'private_basis':private,'private_codes':cpriv}
  payload={**long,**cache};methods.append((name,decoded,payload,long,writes,elapsed,NKEY*(RANK+r)*D*4))
 # Explicit key-value table is the direct retrieval upper control.
 table=values.copy();long={**common,'surprise_write_total':np.asarray([NKEY],dtype=np.uint16),'key_value_table':table};payload={**long,**cache};methods.append(('explicit_key_value_table',table,payload,long,NKEY,0.0,NKEY*D*4))
 rows=[]
 for name,decoded,payload,long,writes,elapsed,mac in methods:
  fullb,longb,cacheb,serwall=save(seed,r,name,payload,long,cache);lmse,aware,nll=evaluate(decoded,values,cache_keys,cache_values)
  rows.append({'seed':seed,'private_value_rank':r,'method':name,'complete_state_bytes':fullb,'longterm_state_bytes':longb,'shortterm_cache_bytes':cacheb,'longterm_query_mse':lmse,'cache_aware_query_mse':aware,'cache_aware_query_nll':nll,'surprise_triggered_writes':writes,'memory_keys':NKEY,'write_events':len(events),'optimizer_updates':writes,'update_mac_proxy':int(mac*2),'query_mac_proxy':int(NKEY*D*(RANK+r) if 'rank4' in name or 'private' in name else NKEY*D*D),'wall_time_s':elapsed,'serialization_wall_s':serwall})
 # Native control gets the exact same coordinates and outcomes.
 a=next(x for x in rows if x['method']=='mirror_shared_private');b=next(x for x in rows if x['method']=='native_lowrank_memory')
 assert a['complete_state_bytes']==b['complete_state_bytes'] and a['cache_aware_query_mse']==b['cache_aware_query_mse']
 return rows

def run(seeds):
 rows=[x for s in seeds for r in (0,2,4,8) for x in run_condition(s,r)]
 (ROOT/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for x in rows:w.writerow([f"private_rank_{x['private_value_rank']}",x['seed'],x['method'],x['complete_state_bytes'],x['write_events'],x['optimizer_updates'],x['update_mac_proxy']+x['query_mac_proxy'],x['wall_time_s'],'cache_aware_query_mse',x['cache_aware_query_mse'],'longterm_query_mse',x['longterm_query_mse'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);a=ap.parse_args();print(json.dumps(run(a.seeds),indent=2))
