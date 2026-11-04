import argparse,csv,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];NL=NR=32;M=256;SCOPE=4;KD=8;D=32;R=4;NABS=256

def norm(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def make_world(seed):
 q=np.random.default_rng(seed);lk=norm(q.normal(size=(NL,KD)).astype('f'));rk=norm(q.normal(size=(NR,KD)).astype('f'));pairs=q.choice(NL*NR,size=M,replace=False);li=pairs//NR;ri=pairs%NR;scopes=q.integers(SCOPE,size=M,dtype=np.uint8);basis=(q.normal(size=(R,D))/D**.5).astype('f');codes=q.normal(size=(M,R)).astype('f');values=codes@basis
 fullkeys=np.concatenate([lk[li],rk[ri]],axis=1);index=np.full((NL,NR),-1,dtype=np.int16);index[li,ri]=np.arange(M,dtype=np.int16)
 active=set(pairs.tolist());absent=np.array([p for p in range(NL*NR) if p not in active]);abs_pairs=q.choice(absent,size=NABS,replace=False);ali=abs_pairs//NR;ari=abs_pairs%NR;asc=q.integers(SCOPE,size=NABS,dtype=np.uint8)
 return lk,rk,li,ri,scopes,basis,codes,values,fullkeys,index,ali,ari,asc

def queries_for(seed,world,sigma):
 lk,rk,li,ri,sc,basis,codes,values,fullkeys,index,ali,ari,asc=world;q=np.random.default_rng(seed+int(sigma*10000)+77)
 left=np.concatenate([lk[li],lk[li],lk[ali]],axis=0);right=np.concatenate([rk[ri],rk[ri],rk[ari]],axis=0);scope=np.concatenate([sc,(sc+1)%SCOPE,asc]);labels=np.concatenate([np.arange(M),np.arange(M),-np.ones(NABS,dtype=int)]);targets=np.concatenate([values,np.zeros_like(values),np.zeros((NABS,D),'f')])
 if sigma:left=norm(left+q.normal(scale=sigma,size=left.shape).astype('f'));right=norm(right+q.normal(scale=sigma,size=right.shape).astype('f'))
 return left,right,scope,labels,targets

def retrieve(method,p,left,right,scope,labels,targets):
 start=time.perf_counter();Q=len(left);pred=np.zeros((Q,D),'f');got=np.full(Q,-1,dtype=int);active=np.zeros(Q,bool)
 if method=='flat_bank':
  scores=np.concatenate([left,right],axis=1)@p['full_keys'].T;idx=np.argmax(scores,axis=1);conf=scores[np.arange(Q),idx]>1.35;ok=conf&(p['scopes'][idx]==scope);got[ok]=idx[ok];active=ok;pred[ok]=p['values'][idx[ok]];mac=Q*M*(2*KD)
 else:
  ls=left@p['left_keys'].T;rs=right@p['right_keys'].T;li=np.argmax(ls,axis=1);ri=np.argmax(rs,axis=1);conf=(ls[np.arange(Q),li]>.55)&(rs[np.arange(Q),ri]>.55);idx=p['index'][li,ri];ok=conf&(idx>=0);rows=np.maximum(idx,0).astype(int);ok &= p['scopes'][rows]==scope;got[ok]=rows[ok];active=ok
  if method=='product_direct':pred[ok]=p['values'][rows[ok]]
  else:pred[ok]=p['codes'][rows[ok]]@p['basis']
  mac=Q*(2*NL*KD+(R*D if method in ('mirror_product','native_lowrank_product') else D))
 wall=time.perf_counter()-start
 positive=labels>=0;wrong=(labels>=0)&(scope!=p['scopes'][np.maximum(labels,0)]) if method=='flat_bank' else (labels>=0)&(scope!=world_scope_placeholder)
 # scope error/false positive evaluated by query group: middle third is wrong-scope.
 group=np.arange(Q)//M; wrong_scope=group==1;absent=labels<0
 recall=float(np.mean(got[:M]==labels[:M]));err=float(np.sqrt(np.mean((pred[:M]-targets[:M])**2)/max(float(np.mean(targets[:M]**2)),1e-12)))
 wrong_rate=float(np.mean(active[wrong_scope]));absent_rate=float(np.mean(active[absent]));collision=float(np.mean((got[:M]>=0)&(got[:M]!=labels[:M])))
 return {'recall':recall,'edit_nrmse':err,'wrong_scope_rate':wrong_rate,'absent_false_activation_rate':absent_rate,'collision_rate':collision,'mac_proxy':int(mac),'wall':wall}

def run(seeds):
 rows=[]
 for seed in seeds:
  world=make_world(seed);lk,rk,li,ri,sc,basis,codes,values,fullkeys,index,ali,ari,asc=world
  payloads={'flat_bank':{'full_keys':fullkeys,'values':values,'scopes':sc,'threshold':np.array([1.35],'f')},'product_direct':{'left_keys':lk,'right_keys':rk,'index':index,'scopes':sc,'values':values,'threshold':np.array([.55],'f')},'mirror_product':{'left_keys':lk,'right_keys':rk,'index':index,'scopes':sc,'basis':basis,'codes':codes,'threshold':np.array([.55],'f')},'native_lowrank_product':{'left_keys':lk.copy(),'right_keys':rk.copy(),'index':index.copy(),'scopes':sc.copy(),'basis':basis.copy(),'codes':codes.copy(),'threshold':np.array([.55],'f')}}
  for sigma in (0.,.25):
   left,right,scope,labels,targets=queries_for(seed,world,sigma)
   for method,p in payloads.items():
    path=ROOT/'source'/f'{seed}_noise{sigma:g}_{method}.npz';st=time.perf_counter();np.savez_compressed(path,**p);serial=time.perf_counter()-st
    # Reopen the serialized payload for authoritative evaluation.
    with np.load(path) as z:
     m=retrieve(method,z,left,right,scope,labels,targets)
    rows.append({'seed':seed,'residual_rms':sigma,'method':method,'serialized_bytes':path.stat().st_size,'output_nrmse':m['edit_nrmse'],'examples':len(left),'optimizer_updates':0,'active_mac_proxy':m['mac_proxy'],'inference_wall_time_s':m['wall'],'serialization_wall_time_s':serial,'retrieval_recall':m['recall'],'wrong_scope_rate':m['wrong_scope_rate'],'absent_false_activation_rate':m['absent_false_activation_rate'],'collision_rate':m['collision_rate']})
 (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','serialization_wall_time_s','status_note'])
  for r in rows:w.writerow([f"noise{r['residual_rms']:g}",r['seed'],r['method'],r['serialized_bytes'],r['examples'],0,r['active_mac_proxy'],r['inference_wall_time_s'],'edit_nrmse',r['output_nrmse'],'retrieval_recall/wrong_scope/absent/collision',f"{r['retrieval_recall']}/{r['wrong_scope_rate']}/{r['absent_false_activation_rate']}/{r['collision_rate']}",r['serialization_wall_time_s'],'development; fresh sealed'])
 return rows
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',nargs='+',type=int,required=True);print(json.dumps(run(a.parse_args().seeds),indent=2))
