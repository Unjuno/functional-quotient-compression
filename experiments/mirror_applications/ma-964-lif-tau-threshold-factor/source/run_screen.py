"""MA-964 fixed recurrent LIF oracle-compression screen."""
import csv,json,struct,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
Q=N=8;T=16;D=8;NTEST=128;FRESH=[96421,96423,96427]
TAU_BOUNDS=(1.5,8.0);THETA_BOUNDS=(.15,1.0)

def teacher_bank(seed,kind):
    rng=np.random.default_rng(seed)
    if kind=='structured_rank1':
        st=rng.uniform(.75,1.25,Q);sn=rng.uniform(2.,6.,N)
        ht=rng.uniform(.72,1.28,Q);hn=rng.uniform(.22,.72,N)
        tau=st[:,None]*sn[None,:];theta=ht[:,None]*hn[None,:]
    else:
        tau=rng.uniform(*TAU_BOUNDS,(Q,N));theta=rng.uniform(*THETA_BOUNDS,(Q,N))
    tau=np.clip(tau,*TAU_BOUNDS);theta=np.clip(theta,*THETA_BOUNDS)
    return tau.astype(np.float32),theta.astype(np.float32)

def factor(m,rank):
    u,s,vh=np.linalg.svd(m.astype(np.float64),full_matrices=False)
    l=u[:,:rank]*np.sqrt(s[:rank])[None,:];r=np.sqrt(s[:rank])[:,None]*vh[:rank]
    return l.astype(np.float32),r.astype(np.float32)

def decode_factors(l,r,bounds):return np.clip(l@r,*bounds).astype(np.float32)

def event_data(seed):
    rng=np.random.default_rng(seed);x=np.empty((Q,NTEST,T,D),np.float32);labels=np.tile([0,1],NTEST//2)
    for q in range(Q):
        y=labels.copy();rng.shuffle(y)
        phase=2*np.pi*(q/Q+np.arange(T)[:,None]/T+np.arange(D)[None,:]/D)
        direction=np.sin(phase)
        p=np.clip(.28+.22*(2*y[:,None,None]-1)*direction[None,:,:],.03,.75)
        x[q]=rng.binomial(1,p).astype(np.float32)
    return x

def synapses(seed):
    rng=np.random.default_rng(seed);wi=rng.normal(0,.55,(N,D)).astype(np.float32)
    wr=rng.normal(0,.05,(N,N)).astype(np.float32);np.fill_diagonal(wr,0)
    eig=np.max(np.abs(np.linalg.eigvals(wr)))
    if eig>0:wr*=.32/eig
    return wi,wr

def simulate(x,wi,wr,tau,theta):
    b=len(x);v=np.zeros((b,N),np.float32);trace=np.zeros((b,T,N),np.uint8);maxv=0.
    alpha=np.exp(-1./tau).astype(np.float32)
    for t in range(T):
        current=x[:,t,:]@wi.T+trace[:,t-1,:].astype(np.float32)@wr.T if t else x[:,0,:]@wi.T
        v=alpha*v+current
        maxv=max(maxv,float(np.max(np.abs(v))))
        sp=(v>=theta).astype(np.uint8);trace[:,t]=sp
        v=v-sp.astype(np.float32)*theta
    return trace,maxv

def pack_state(meta,arrays):
    blob=[];items=[];offset=0
    for name,a in arrays.items():
        v=np.ascontiguousarray(a);raw=v.tobytes();items.append({'name':name,'shape':list(v.shape),'dtype':v.dtype.str,'offset':offset,'nbytes':len(raw)})
        blob.append(raw);offset+=len(raw)
    payload=b''.join(blob);head=json.dumps({'meta':meta,'tensors':items,'payload_bytes':len(payload)},separators=(',',':')).encode()
    return b'MA964v1\0'+struct.pack('<I',len(head))+head+payload

def unpack(blob):
    assert blob[:8]==b'MA964v1\0';n=struct.unpack('<I',blob[8:12])[0];h=json.loads(blob[12:12+n]);payload=memoryview(blob)[12+n:];out={}
    for x in h['tensors']:
        a=np.frombuffer(payload[x['offset']:x['offset']+x['nbytes']],dtype=np.dtype(x['dtype'])).copy().reshape(x['shape']);out[x['name']]=a
    return h['meta'],out

def state_for(condition,wi,wr,tau,theta):
    s={'encoder':wi,'recurrent':wr}
    if condition=='shared':s.update(tau_shared=tau.mean(0),theta_shared=theta.mean(0))
    elif condition=='full':s.update(tau=tau,theta=theta)
    else:
        rank=int(condition[-1]);tl,tr=factor(tau,rank);hl,hr=factor(theta,rank)
        s.update(tau_left=tl,tau_right=tr,theta_left=hl,theta_right=hr)
    return s

def reconstruct(condition,s):
    if condition=='shared':return np.broadcast_to(s['tau_shared'],(Q,N)).copy(),np.broadcast_to(s['theta_shared'],(Q,N)).copy(),0
    if condition=='full':return s['tau'],s['theta'],0
    rank=int(condition[-1]);return decode_factors(s['tau_left'],s['tau_right'],TAU_BOUNDS),decode_factors(s['theta_left'],s['theta_right'],THETA_BOUNDS),2*Q*N*rank

def compare(pred,ref):
    p=pred.astype(bool).reshape(-1,T,N);r=ref.astype(bool).reshape(-1,T,N);tp=np.logical_and(p,r).sum();fp=np.logical_and(p,~r).sum();fn=np.logical_and(~p,r).sum()
    f1=2*tp/max(1,2*tp+fp+fn);agreement=float(np.mean(p==r));spikes=float(p.sum()/len(p));
    lat=[]
    for a,b in zip(p,r):
        ia=np.flatnonzero(a.any(1));ib=np.flatnonzero(b.any(1));la=int(ia[0]+1) if len(ia) else T;lb=int(ib[0]+1) if len(ib) else T;lat.append(abs(la-lb))
    return agreement,float(f1),spikes,float(np.mean(lat))

def run(seed,kind):
    tau,theta=teacher_bank(seed+10,kind);wi,wr=synapses(seed+20);x=event_data(seed+30);rows=[];blobs={};teacher=[]
    for q in range(Q):teacher.append(simulate(x[q],wi,wr,tau[q],theta[q])[0])
    teacher=np.stack(teacher)
    conds=['shared','full','mirror_rank1','native_rank1','mirror_rank2','native_rank2']
    for c in conds:
        # Use exactly the same factor extraction for Mirror and native parameterization.
        alias='rank1' if c=='mirror_rank1' else 'rank2' if c=='mirror_rank2' else c
        raw=state_for(alias,wi,wr,tau,theta);meta={'condition':c,'bank':kind,'seed':seed,'tasks':Q,'neurons':N,'timesteps':T,'task_ids':'row index; charged in manifest'}
        blob=pack_state(meta,raw);loaded_meta,s=unpack(blob);blobs[c]=blob
        state_roundtrip_exact=all(np.array_equal(raw[k],s[k]) for k in raw)
        assert state_roundtrip_exact
        assert all(np.array_equal(raw[k],s[k]) for k in raw)
        wi2=s['encoder'];wr2=s['recurrent'];tt,th,decode_macs=reconstruct(alias,s)
        outputs=[];stability=0.
        for q in range(Q):
            out,mx=simulate(x[q],wi2,wr2,tt[q],th[q]);outputs.append(out);stability=max(stability,mx)
        outputs=np.stack(outputs);agreement,f1,spikes,lat=compare(outputs,teacher)
        start=time.perf_counter()
        for _ in range(5):
            for q in range(Q):simulate(x[q],wi2,wr2,tt[q],th[q])
        infer_s=(time.perf_counter()-start)/5
        rows.append({'seed':seed,'family':kind,'condition':c,'payload_bytes':len(blob),'spike_hamming_agreement':agreement,
          'spike_f1':f1,'mean_spikes_per_sequence':spikes,'mean_first_spike_latency_abs_error':lat,'max_abs_membrane':stability,
          'neuron_time_updates_per_sequence':N*T,'fixed_synapse_mac_proxy_per_sequence':T*N*(D+N),
          'one_time_factor_decode_macs':decode_macs,'dense_tau_theta_runtime_bytes':Q*N*2*4,
          'inference_sequences_per_second':Q*NTEST/infer_s,'inference_seconds_per_repeat':infer_s,
          'serialization_state_roundtrip_exact':state_roundtrip_exact})
        print(json.dumps({k:rows[-1][k] for k in ['seed','family','condition','payload_bytes','spike_hamming_agreement','spike_f1','mean_spikes_per_sequence','max_abs_membrane']}),flush=True)
    for r in [1,2]:
        a=blobs[f'mirror_rank{r}'];b=blobs[f'native_rank{r}'];_,sa=unpack(a);_,sb=unpack(b)
        assert all(np.array_equal(sa[k],sb[k]) for k in sa)
        ra=next(v for v in rows if v['condition']==f'mirror_rank{r}');rb=next(v for v in rows if v['condition']==f'native_rank{r}')
        assert ra['spike_hamming_agreement']==rb['spike_hamming_agreement'] and ra['payload_bytes']==rb['payload_bytes']
    return rows

def main():
    rows=[]
    for seed in FRESH:
        for family in ['structured_rank1','generic_random']:
            rows.extend(run(seed,family))
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'source/results.json').write_text(json.dumps({'results':rows,'fresh_seeds':FRESH,'families':['structured_rank1','generic_random'],'device':'cpu'},indent=2)+'\n')
if __name__=='__main__':main()
