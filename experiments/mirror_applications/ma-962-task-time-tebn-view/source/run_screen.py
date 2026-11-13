"""MA-962 custom CPU LIF and task/time normalization experiment."""
import argparse,csv,json,struct,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
Q=8;T=16;CH=8;H=2;RANK=4;LEAK=.9;THRESHOLD=1.;UPDATES=1000;BATCH=128;LR=.02
FRESH=[96211,96213,96217];DEV=96201;torch.set_num_threads(1)
GENERIC_FRESH=[96241,96243,96247]

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):ctx.save_for_backward(x);return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx,grad):
        (x,)=ctx.saved_tensors
        return grad*torch.clamp(1.-torch.abs(x),min=0.)

def signature(kind,seed):
    if kind=='structured_rank3':
        q=np.arange(Q)[:,None]/Q;t=np.arange(T)[None,:]/T
        return .35+.65*np.cos(2*np.pi*(q-t))
    rng=np.random.default_rng(seed+3001);k=rng.uniform(-1.,1.,(Q,T));return k/np.abs(k).max()

def make_data(seed,n_per_task,kind,bank_seed=None):
    rng=np.random.default_rng(seed);k=signature(kind,seed if bank_seed is None else bank_seed);xs=[];ys=[];qs=[]
    for q in range(Q):
        y=np.tile(np.array([0,1],dtype=np.int64),(n_per_task+1)//2)[:n_per_task]
        rng.shuffle(y);sign=(2*y-1)[:,None];signal=sign*k[q][None,:]
        ppos=np.clip(.5+.4*signal,0.,1.);pneg=np.clip(.5-.4*signal,0.,1.)
        pos=rng.binomial(1,ppos[:,:,None],size=(n_per_task,T,4))
        neg=rng.binomial(1,pneg[:,:,None],size=(n_per_task,T,4))
        xs.append(np.concatenate([pos,neg],axis=-1).astype('float32'));ys.append(y);qs.append(np.full(n_per_task,q,dtype=np.int64))
    return (torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys)),
            torch.from_numpy(np.concatenate(qs)),k)

class TemporalSNN(nn.Module):
    def __init__(self,condition):
        super().__init__();self.condition=condition
        w=torch.cat([torch.full((4,),.25),torch.full((4,),-.25)])
        self.register_buffer('encoder',torch.stack([w,-w]))
        # Keep shared batch statistics but freeze affine freedom so it cannot
        # absorb or cancel the task/time gain effect under test.
        self.bn=nn.BatchNorm1d(H,affine=False)
        if condition=='shared_tebn':self.shared_gain=nn.Parameter(torch.ones(T))
        elif condition=='full_tebn':self.full_gain=nn.Parameter(torch.ones(Q,T))
        elif condition=='rank1_native':self.task_scale=nn.Parameter(torch.ones(Q));self.time_basis=nn.Parameter(torch.ones(T))
        elif condition in ('mirror_rank4','native_rank4'):
            u=torch.zeros(Q,RANK);v=torch.zeros(RANK,T);u[:,0]=1.;v[0]=1.
            torch.nn.init.normal_(u[:,1:],std=.03);torch.nn.init.normal_(v[1:],std=.03)
            self.task_code=nn.Parameter(u);self.time_basis=nn.Parameter(v)
        else:raise ValueError(condition)
    def gain_matrix(self):
        if self.condition=='shared_tebn':return self.shared_gain[None,:].expand(Q,-1)
        if self.condition=='full_tebn':return self.full_gain
        if self.condition=='rank1_native':return self.task_scale[:,None]*self.time_basis[None,:]
        return self.task_code@self.time_basis
    def forward(self,x,q,return_trace=False):
        b=x.shape[0];z=F.linear(x,self.encoder);z=self.bn(z.transpose(1,2)).transpose(1,2)
        g=self.gain_matrix()[q];v=x.new_zeros((b,H));tr=[]
        for t in range(T):
            v=LEAK*v+g[:,t,None]*z[:,t]
            s=SurrogateSpike.apply(v-THRESHOLD)
            tr.append(s);v=v-s.detach()
        trace=torch.stack(tr,dim=1);counts=trace.sum(1)
        return (counts,trace) if return_trace else counts

def pack_state(model,meta):
    arrays=[];items=[];offset=0
    for name,tensor in model.state_dict().items():
        a=tensor.detach().cpu().contiguous().numpy();raw=a.tobytes(order='C')
        items.append({'name':name,'shape':list(a.shape),'dtype':a.dtype.str,'offset':offset,'nbytes':len(raw)})
        arrays.append(raw);offset+=len(raw)
    data=b''.join(arrays);header=json.dumps({'metadata':meta,'tensors':items,'tensor_payload_bytes':len(data)},separators=(',',':')).encode()
    return b'MA962v1\0'+struct.pack('<I',len(header))+header+data

def unpack_state(blob):
    assert blob[:8]==b'MA962v1\0';n=struct.unpack('<I',blob[8:12])[0];h=json.loads(blob[12:12+n]);data=memoryview(blob)[12+n:];state={}
    for it in h['tensors']:
        raw=data[it['offset']:it['offset']+it['nbytes']];a=np.frombuffer(raw,dtype=np.dtype(it['dtype'])).copy().reshape(it['shape'])
        state[it['name']]=torch.from_numpy(a)
    return h['metadata'],state

def evaluate(model,x,y,q,batch=BATCH):
    model.eval();all_counts=[];all_trace=[];t0=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(y),batch):
            c,tr=model(x[i:i+batch],q[i:i+batch],True);all_counts.append(c);all_trace.append(tr)
    elapsed=time.perf_counter()-t0;counts=torch.cat(all_counts);trace=torch.cat(all_trace)
    pred=counts.argmax(1);acc_by=[];lat_by=[]
    prefix=trace.cumsum(1).argmax(2);anyspike=trace.sum(-1).cumsum(1)>0
    correct=(prefix==y[:,None])&anyspike;first=correct.float().argmax(1)+1
    anycorrect=correct.any(1);lat=torch.where(anycorrect,first,torch.full_like(first,T))
    for task in range(Q):
        ix=q==task;acc_by.append(float((pred[ix]==y[ix]).float().mean()))
        lat_by.append(float(lat[ix].float().mean()))
    return {'accuracy':float((pred==y).float().mean()),'accuracy_by_task':acc_by,
            'mean_spikes_per_example':float(counts.sum(1).mean()),'spikes_by_task':[float(counts[q==i].sum(1).mean()) for i in range(Q)],
            'first_correct_latency_by_task':lat_by,'inference_examples_per_second':len(y)/elapsed,
            'inference_seconds':elapsed,'max_spikes':int(counts.max())}

def fit(condition,seed,kind,train,test,updates=UPDATES):
    xtr,ytr,qtr,_=train;xte,yte,qte,_=test
    torch.manual_seed(seed+808);m=TemporalSNN(condition);opt=torch.optim.Adam(m.parameters(),lr=LR)
    rng=torch.Generator().manual_seed(seed+1777);t0=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(ytr),(BATCH,),generator=rng)
        logits=m(xtr[ix],qtr[ix]);loss=F.cross_entropy(logits,ytr[ix]);opt.zero_grad();loss.backward();opt.step()
    train_seconds=time.perf_counter()-t0;m.eval()
    meta={'condition':condition,'family':kind,'seed':seed,'tasks':Q,'timesteps':T,'lif_leak':LEAK,'threshold':THRESHOLD}
    blob=pack_state(m,meta);_,state=unpack_state(blob);clone=TemporalSNN(condition);clone.load_state_dict(state);clone.eval()
    with torch.no_grad():replay=float((m(xte[:256],qte[:256])-clone(xte[:256],qte[:256])).abs().max())
    metrics=evaluate(clone,xte,yte,qte)
    param_count=sum(v.numel() for v in m.parameters())
    if condition in ('mirror_rank4','native_rank4'):gain_ops_per_example=Q*0+RANK*T
    elif condition=='rank1_native':gain_ops_per_example=T
    elif condition=='full_tebn':gain_ops_per_example=0
    else:gain_ops_per_example=0
    row={'seed':seed,'family':kind,'condition':condition,'payload_bytes':len(blob),'train_examples_seen':updates*BATCH,
         'updates':updates,'train_seconds':train_seconds,'trainable_parameters':param_count,
         'lif_neuron_updates_per_example':2*T,'gain_factor_multiplies_per_example':gain_ops_per_example,
         'serialization_replay_max_abs':replay,**metrics}
    return row,blob

def run_seed(seed,kind,updates,conditions):
    bank_seed=seed+3001
    train=make_data(seed+100,256,kind,bank_seed);test=make_data(seed+200,512,kind,bank_seed);rows=[];blobs={}
    for condition in conditions:
        row,blob=fit(condition,seed,kind,train,test,updates);rows.append(row);blobs[condition]=blob
        print(json.dumps({k:row[k] for k in ['seed','family','condition','payload_bytes','accuracy','mean_spikes_per_example','train_seconds','inference_examples_per_second','serialization_replay_max_abs']}),flush=True)
    if 'mirror_rank4' in blobs and 'native_rank4' in blobs:
        # Identical rank-r equations and seed/data/update schedule are an algebraic alias control.
        assert rows[[r['condition'] for r in rows].index('mirror_rank4')]['accuracy_by_task']==rows[[r['condition'] for r in rows].index('native_rank4')]['accuracy_by_task']
        assert rows[[r['condition'] for r in rows].index('mirror_rank4')]['payload_bytes']==rows[[r['condition'] for r in rows].index('native_rank4')]['payload_bytes']
        _,sm=unpack_state(blobs['mirror_rank4']);_,sn=unpack_state(blobs['native_rank4'])
        assert all(torch.equal(sm[k],sn[k]) for k in sm)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dev',action='store_true');ap.add_argument('--generic-fresh',action='store_true');a=ap.parse_args()
    if a.dev:
        rows=[]
        for family in ['structured_rank3','generic_random']:
            rows+=run_seed(DEV,family,UPDATES,['full_tebn','mirror_rank4','native_rank4'])
        (ROOT/'source/development.json').write_text(json.dumps(rows,indent=2)+'\n');return
    if a.generic_fresh:
        rows=[]
        for seed in GENERIC_FRESH:
            rows+=run_seed(seed,'generic_random',UPDATES,['shared_tebn','full_tebn','rank1_native','mirror_rank4','native_rank4'])
        with (ROOT/'RESULTS_GENERIC.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (ROOT/'source/generic_results.json').write_text(json.dumps({'results':rows,'fresh_seeds':GENERIC_FRESH,'family':'generic_random','device':'cpu'},indent=2)+'\n');return
    rows=[]
    for seed in FRESH:
        rows+=run_seed(seed,'structured_rank3',UPDATES,['shared_tebn','full_tebn','rank1_native','mirror_rank4','native_rank4'])
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'source/results.json').write_text(json.dumps({'results':rows,'fresh_seeds':FRESH,'families':['structured_rank3','generic_random'],
      'device':'cpu','torch':torch.__version__,'updates':UPDATES,'batch_size':BATCH,'lif_steps':T,'tasks':Q},indent=2)+'\n')
if __name__=='__main__':main()
