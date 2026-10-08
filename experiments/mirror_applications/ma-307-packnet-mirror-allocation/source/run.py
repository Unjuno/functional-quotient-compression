import argparse,csv,json,struct,time
from pathlib import Path
import numpy as np
T,H,DX,ALIGNED=8,32,16,5
METHODS=['tied','packnet','mirror','independent']
SCALE=0.2/127

def rotate(v,theta):
    out=np.empty_like(v);omega=np.arange(1,H//2+1)
    a=theta*omega;c=np.cos(a);s=np.sin(a)
    x=v[0::2];y=v[1::2]
    out[0::2]=c*x-s*y;out[1::2]=s*x+c*y
    return out

def world(seed):
    rng=np.random.default_rng(seed);enc=rng.normal(size=(DX,H))/np.sqrt(DX);base=rng.normal(size=H)/np.sqrt(H)
    theta=np.r_[0.,rng.uniform(-.12,.12,ALIGNED-1)]
    targets=[rotate(base,a) for a in theta]
    targets += [rng.normal(size=H)/np.sqrt(H) for _ in range(T-ALIGNED)]
    data=[]
    for i,w in enumerate(targets):
        blocks=[]
        for n,salt in [(512,100),(256,200),(512,300)]:
            x=np.random.default_rng(seed+salt+i*17).normal(size=(n,DX));h=np.maximum(0,x@enc);y=h@w
            blocks.append((h,y))
        data.append(blocks)
    return enc,base,data

def fit_ls(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]

def n_mse(h,y,w):return float(np.mean((h@w-y)**2)/(np.mean(y*y)+1e-12))

def encode(method,base,records,privates,seed):
    cfg={'method':method,'tasks':T,'hidden':H,'input':DX,'encoder_seed':seed,'frequency_rule':'pairwise integer frequencies 1..16','records':records,'has_base':method!='independent'}
    payload=bytearray()
    if method!='independent':payload.extend(np.asarray(base,dtype='<f4').tobytes())
    for i,r in enumerate(records):
        if r['kind']=='view':payload.extend(np.asarray([r['code']],dtype='i1').tobytes())
        elif r['kind']=='mask':payload.extend(np.packbits(np.asarray(r['mask'],dtype=np.uint8),bitorder='little').tobytes())
        elif r['kind']=='private':payload.extend(np.asarray(privates[i],dtype='<f4').tobytes())
    header=json.dumps(cfg,separators=(',',':'),sort_keys=True).encode()
    return struct.pack('<H',len(header))+header+bytes(payload)

def decode(blob):
    n=struct.unpack('<H',blob[:2])[0];cfg=json.loads(blob[2:2+n]);buf=memoryview(blob)[2+n:];off=0
    def take(dtype,count):
        nonlocal off
        v=np.frombuffer(buf,dtype=dtype,count=count,offset=off).copy();off+=v.nbytes;return v
    base=take('<f4',H) if cfg['has_base'] else None;vectors=[];privates=[]
    for i,r in enumerate(cfg['records']):
        if r['kind']=='view':vectors.append(rotate(base,float(take('i1',1)[0])*SCALE))
        elif r['kind']=='mask':
            mask=np.unpackbits(take('u1',H//8),bitorder='little')[:H];vectors.append(base*mask)
        elif r['kind']=='private':
            v=take('<f4',H);vectors.append(v);privates.append(v)
        else:vectors.append(base)
    return np.stack(vectors),cfg

def mirror_candidate(base,h,y):
    grid=np.linspace(-.2,.2,401);best=(float('inf'),0.)
    for a in grid:
        q=rotate(base,a);loss=np.mean((h@q-y)**2)
        if loss<best[0]:best=(loss,a)
    code=int(np.clip(np.rint(best[1]/SCALE),-127,127));return code,rotate(base,code*SCALE),len(grid)*len(y)*H

def mask_candidate(base,h,y):
    best=(float('inf'),None);ops=0
    for initial in [0,1]:
        mask=np.full(H,initial,dtype=np.uint8)
        for _ in range(4):
            changed=False
            for j in range(H):
                trial=mask.copy();trial[j]^=1;loss=np.mean((h@(base*trial)-y)**2);ops+=len(y)*H
                current=np.mean((h@(base*mask)-y)**2)
                if loss+1e-12<current:mask=trial;changed=True
            if not changed:break
        loss=np.mean((h@(base*mask)-y)**2)
        if loss<best[0]:best=(loss,mask.copy())
    return best[1],base*best[1],ops

def evaluate_stream(method,seed,data,threshold):
    started=time.time();base=fit_ls(*data[0][0]);records=[];privates={};events=[];search_ops=0;prev_errors=[]
    for task,(train,val,test) in enumerate(data):
        prior_before=[]
        if records:
            before,_=decode(encode(method,base,records,privates,seed))
            prior_before=[n_mse(*data[j][2],before[j]) for j in range(task)]
        if method=='independent':
            w=fit_ls(*train);records.append({'kind':'private'});privates[task]=w;choice='private';alloc=True;valerr=n_mse(*val,w);ops=len(train[1])*H*H
        elif task==0:
            records.append({'kind':'shared'});choice='shared_base';alloc=False;valerr=n_mse(*val,base);w=base;ops=len(train[1])*H*H
        elif method=='tied':
            records.append({'kind':'shared'});choice='hard_tie';alloc=False;w=base;valerr=n_mse(*val,w);ops=0
        elif method=='mirror':
            code,w,ops=mirror_candidate(base,*train);search_ops+=ops;valerr=n_mse(*val,w)
            if valerr<=threshold:records.append({'kind':'view','code':code});choice='mirror_view';alloc=False
            else:w=fit_ls(*train);records.append({'kind':'private'});privates[task]=w;choice='private_fallback';alloc=True;ops+=len(train[1])*H*H
        else:
            mask,w,ops=mask_candidate(base,*train);search_ops+=ops;valerr=n_mse(*val,w)
            if valerr<=threshold:records.append({'kind':'mask','mask':mask.tolist()});choice='binary_mask';alloc=False
            else:w=fit_ls(*train);records.append({'kind':'private'});privates[task]=w;choice='private_allocation';alloc=True;ops+=len(train[1])*H*H
        blob=encode(method,base,records,privates,seed);decoded,_=decode(blob)
        olderr=0.;retention_change=0.
        if task:
            diffs=[]
            for old in range(task):
                htest,ytest=data[old][2];diffs.append(n_mse(htest,ytest,decoded[old]))
            olderr=max(diffs);retention_change=max(abs(a-b) for a,b in zip(diffs,prior_before))
        test_h,test_y=test;taskerr=n_mse(test_h,test_y,decoded[task]);prev_errors.append(olderr)
        events.append({'world':seed,'method':method,'task_index':task,'task_kind':'aligned' if task<ALIGNED else 'unrelated','choice':choice,'allocated_private':alloc,'validation_n_mse':valerr,'test_n_mse':taskerr,'incremental_payload_bytes':len(blob)-(events[-1]['cumulative_payload_bytes'] if events else 0),'cumulative_payload_bytes':len(blob),'max_prior_task_n_mse_after_addition':olderr,'max_prior_task_n_mse_change':retention_change,'search_ops_proxy':ops})
    elapsed=time.time()-started;blob=encode(method,base,records,privates,seed);vectors,_=decode(blob)
    errs=[n_mse(*data[i][2],vectors[i]) for i in range(T)];meanerr=float(np.mean(errs));priv_count=sum(r['kind']=='private' for r in records)
    # Includes function-vector decoding/materialization once per active task and held-out matrix evaluation.
    t0=time.time()
    for _ in range(5):
        vv,_=decode(blob)
        for i in range(T):data[i][2][0]@vv[i]
    throughput=5*sum(len(d[2][1]) for d in data)/(time.time()-t0)
    summary={'condition':'fresh' if seed>=30712 else 'development','world_or_seed':seed,'method':method,'serialized_bytes':len(blob),'train_tokens_or_examples':T*512,'optimizer_updates':0,'active_compute_proxy':search_ops+T*512*H*H,'wall_time_s':elapsed,'primary_metric':'mean_normalized_test_mse','primary_value':meanerr,'secondary_metric':'precomputed_task_vectors_per_s','secondary_value':throughput,'status_note':f'private_allocations={priv_count}; accepted_views={sum(r["kind"]=="view" for r in records)}; search_ops={search_ops}; per_task_mse={errs}'}
    return summary,events,priv_count

def run(seed,threshold):
    _,_,data0=world(seed);summaries=[];events=[]
    for m in METHODS:
        s,e,n=evaluate_stream(m,seed,data0,threshold);summaries.append(s);events.extend(e)
    return summaries,events

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--out',required=True);p.add_argument('--events',required=True);a=p.parse_args();ss,ee=run(a.world,a.threshold);Path(a.out).write_text(json.dumps(ss,indent=2));Path(a.events).write_text(json.dumps(ee,indent=2));
    for x in ss:print(x['world_or_seed'],x['method'],f"nMSE={x['primary_value']:.6g}",f"bytes={x['serialized_bytes']}",f"priv={x['status_note'].split(';')[0]}")
