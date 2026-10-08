import argparse, hashlib, io, json, math, sys, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
N_LAYER, N_HEAD, N_EMBD, BLOCK, BATCH, STEPS = 4, 4, 64, 64, 32, 500
LR = 1e-3
FAMILIES = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'

def get_text(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): urllib.request.urlretrieve(URL,path)
    raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    if digest != SHA256: raise ValueError(f'dataset hash mismatch: {digest}')
    text=raw.decode('utf-8'); vocab=sorted(set(text)); stoi={c:i for i,c in enumerate(vocab)}
    ids=np.asarray([stoi[c] for c in text],dtype=np.int64); n=len(ids)
    a=int(.8*n); b=int(.9*n)
    return ids[:a],ids[a:b],ids[b:],vocab,digest

def new_model(seed,vocab_size):
    sys.path.insert(0,str(REPO/'third_party/nanoGPT'))
    from model import GPT,GPTConfig
    torch.manual_seed(seed);torch.set_num_threads(1)
    cfg=GPTConfig(block_size=BLOCK,vocab_size=vocab_size,n_layer=N_LAYER,n_head=N_HEAD,n_embd=N_EMBD,dropout=0.0,bias=True)
    return GPT(cfg)

def batch(ids,rng):
    starts=rng.integers(0,len(ids)-BLOCK-1,size=BATCH)
    x=np.stack([ids[i:i+BLOCK] for i in starts]); y=np.stack([ids[i+1:i+BLOCK+1] for i in starts])
    return torch.from_numpy(x),torch.from_numpy(y)

def train(seed,train_ids,dev_ids,vocab_size):
    model=new_model(seed,vocab_size);model.train();opt=torch.optim.AdamW(model.parameters(),lr=LR,betas=(.9,.99),weight_decay=.01)
    rng=np.random.default_rng(seed+8801); t0=time.perf_counter(); token_count=0
    for step in range(STEPS):
        x,y=batch(train_ids,rng);token_count+=x.numel();opt.zero_grad(set_to_none=True);_,loss=model(x,y);loss.backward();opt.step()
    wall=time.perf_counter()-t0;model.eval()
    return model,{'steps':STEPS,'tokens_seen':token_count,'train_wall_s':wall,'dev_nll':evaluate(model,dev_ids)}

@torch.no_grad()
def evaluate(model,ids):
    model.eval(); total=0.;count=0;t0=time.perf_counter();batch_size=16
    for start in range(0,len(ids)-1,BLOCK*batch_size):
        xs=[];ys=[]
        for j in range(start,min(start+BLOCK*batch_size,len(ids)-1),BLOCK):
            end=min(j+BLOCK,len(ids)-1)
            if end-j<2: continue
            xs.append(ids[j:end]);ys.append(ids[j+1:end+1])
        if not xs:continue
        maxlen=max(map(len,xs));x=np.zeros((len(xs),maxlen),dtype=np.int64);y=np.full((len(xs),maxlen),-1,dtype=np.int64)
        for i,(xx,yy) in enumerate(zip(xs,ys)):x[i,:len(xx)]=xx;y[i,:len(yy)]=yy
        logits,_=model(torch.from_numpy(x),torch.from_numpy(y)); total+=float(F.cross_entropy(logits.reshape(-1,logits.size(-1)),torch.from_numpy(y).reshape(-1),reduction='sum',ignore_index=-1));count+=sum(len(v) for v in ys)
    return {'nll':total/count,'tokens':count,'wall_s':time.perf_counter()-t0,'tokens_per_s':count/max(time.perf_counter()-t0,1e-12)}

def tensors(model):return {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}

def compress(state,method):
    payload={'format':'ma319-v1','method':method,'config':{'n_layer':N_LAYER,'n_head':N_HEAD,'n_embd':N_EMBD,'block_size':BLOCK},'tensors':{},'banks':{}}
    targets={f:[f'transformer.h.{i}.{f}' for i in range(N_LAYER)] for f in FAMILIES}
    keyset={k for ks in targets.values() for k in ks}
    for k,v in state.items():
        if k not in keyset:payload['tensors'][k]=v.to(torch.float32).contiguous()
    fitops=0;reconerr=[]
    for fam,keys in targets.items():
        mats=torch.stack([state[k].float().reshape(-1) for k in keys]).numpy();mean=mats.mean(0);centered=mats-mean
        u,s,vh=np.linalg.svd(centered,full_matrices=False)
        rank= {'tucker1':1,'tucker2':2,'tucker4':4}.get(method,2)
        basis=vh[:rank].astype(np.float32);coeff=centered@basis.T
        if method=='mirror':
            c0=coeff.mean(0);delta=coeff-c0;radius=float(np.mean(np.linalg.norm(delta,axis=1)))
            if radius<1e-12: angles=np.zeros(N_LAYER,dtype=np.float32)
            else: angles=np.arctan2(delta[:,1],delta[:,0]).astype(np.float32)
            code={'center':torch.tensor(c0,dtype=torch.float16),'radius':torch.tensor([radius],dtype=torch.float16),'angles':torch.tensor(angles,dtype=torch.float16)}
            decoded=np.stack([c0+radius*np.array([np.cos(float(a)),np.sin(float(a))]) for a in angles])
        else:
            code={'coeff':torch.tensor(coeff,dtype=torch.float16)}
            decoded=code['coeff'].float().numpy()
        rec=mean[None,:]+decoded@basis
        recerr=float(np.mean((mats-rec)**2));reconerr.append(recerr)
        payload['banks'][fam]={'shape':list(state[keys[0]].shape),'mean':torch.tensor(mean,dtype=torch.float16),'basis':torch.tensor(basis,dtype=torch.float16),**code}
        fitops+=2*N_LAYER*rank*len(mean)
    payload['compression_fit_ops_proxy']=int(fitops)
    payload['mean_matrix_reconstruction_mse']=float(np.mean(reconerr))
    return payload

def decode(payload,state_template):
    out={k:v.clone().float() for k,v in payload['tensors'].items()}
    for fam,bank in payload['banks'].items():
        shape=tuple(bank['shape']);mean=bank['mean'].float().numpy();basis=bank['basis'].float().numpy()
        if payload['method']=='mirror':
            c=bank['center'].float().numpy();r=float(bank['radius'].float()[0]);angles=bank['angles'].float().numpy()
            coeff=np.stack([c+r*np.array([np.cos(float(a)),np.sin(float(a))]) for a in angles])
        else: coeff=bank['coeff'].float().numpy()
        matrices=mean[None,:]+coeff@basis
        for i,vec in enumerate(matrices):out[f'transformer.h.{i}.{fam}']=torch.from_numpy(vec.reshape(shape).copy()).float()
    return out

def build_model(seed,vocab_size,state):
    m=new_model(seed,vocab_size);m.load_state_dict(state);m.eval();return m

def serialize(payload):
    bio=io.BytesIO();torch.save(payload,bio);return bio.getvalue()

def run(seed,split,datapath,outdir):
    train_ids,dev_ids,fresh_ids,vocab,dhash=get_text(datapath);model,train_stats=train(seed,train_ids,dev_ids,len(vocab));base=tensors(model);rows=[];payload_dir=Path(outdir);payload_dir.mkdir(parents=True,exist_ok=True)
    for method in ['full','tucker1','tucker2','tucker4','mirror']:
        t0=time.perf_counter()
        if method=='full':payload={'format':'ma319-v1','method':'full','config':{'n_layer':N_LAYER,'n_head':N_HEAD,'n_embd':N_EMBD,'block_size':BLOCK},'tensors':{k:v.float() for k,v in base.items()},'banks':{}}
        else:payload=compress(base,method)
        fit_wall=time.perf_counter()-t0;data=serialize(payload);path=payload_dir/f'{split}_{seed}_{method}.pt';path.write_bytes(data)
        t1=time.perf_counter();decoded=payload['tensors'] if method=='full' else decode(payload,base);decode_wall=time.perf_counter()-t1
        candidate=build_model(seed,len(vocab),decoded);fresh=evaluate(candidate,fresh_ids)
        dev=evaluate(candidate,dev_ids)
        rows.append({'condition':split,'seed':seed,'method':method,'payload_bytes':len(data),'payload_sha256':hashlib.sha256(data).hexdigest(),'train_steps':STEPS,'train_tokens':train_stats['tokens_seen'],'train_wall_s':train_stats['train_wall_s'],'base_dev_nll':train_stats['dev_nll']['nll'],'dev_nll':dev['nll'],'fresh_nll':fresh['nll'],'fresh_tokens':fresh['tokens'],'fresh_eval_wall_s':fresh['wall_s'],'fresh_tokens_per_s':fresh['tokens_per_s'],'compression_wall_s':fit_wall,'decode_wall_s':decode_wall,'compression_fit_ops_proxy':payload.get('compression_fit_ops_proxy',0),'matrix_reconstruction_mse':payload.get('mean_matrix_reconstruction_mse',0.0)})
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--data',default=str(ROOT/'source/data/tinyshakespeare.txt'));ap.add_argument('--outdir',default=str(ROOT/'artifacts/packages'));ap.add_argument('--json',required=True);a=ap.parse_args();rows=run(a.seed,a.split,a.data,a.outdir);p=Path(a.json);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps({'dataset_sha256':SHA256,'rows':rows},indent=2)+'\n')
