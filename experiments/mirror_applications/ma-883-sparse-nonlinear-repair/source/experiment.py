"""MA-883 pooled linear cross-model KV map with sparse private MLP residuals."""
from __future__ import annotations
import os
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import hashlib,json,resource,re,struct,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache

SOURCES=('EleutherAI/pythia-70m-deduped','EleutherAI/pythia-160m-deduped')
TARGET='HuggingFaceTB/SmolLM2-135M'; PAIRS=tuple((s,TARGET) for s in SOURCES); MODELS=SOURCES+(TARGET,)
D=64; LAYERS=30; HEADS=3; ROLES=LAYERS*HEADS*2; ALPHA=.01
HIDDEN_CANDIDATES=(4,8,16); FRESH_SEEDS=(983,997,1009); DEV_BUCKET=0; UPDATE_STEPS=24; LR=.001
MIN_TOKENS,MAX_TOKENS=16,128; DATASET_NAME='Salesforce/wikitext'; DATASET_CONFIG='wikitext-103-raw-v1'

def snapshot(mid:str)->Path:
    p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots'
    return sorted(p.iterdir())[0]

def load_assets():
    torch.set_num_threads(min(4,os.cpu_count() or 1)); models={}; toks={}; paths={}
    for mid in MODELS:
        p=snapshot(mid); paths[mid]=p; toks[mid]=AutoTokenizer.from_pretrained(p,local_files_only=True,use_fast=True)
        models[mid]=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,attn_implementation='eager').eval()
    cfg=models[TARGET].config
    if (cfg.num_hidden_layers,getattr(cfg,'num_key_value_heads',None) or cfg.num_attention_heads)!=(LAYERS,HEADS): raise ValueError('unexpected SmolLM2 config')
    for p in MODELS:
        for param in models[p].parameters(): param.requires_grad_(False)
    return models,toks,paths

def prompt_manifest()->dict[str,Any]: return json.loads((Path(__file__).resolve().parents[1]/'prompt_manifest.json').read_text())

def load_prompt_sets(toks:dict[str,Any])->dict[str,dict[str,list[dict[str,Any]]]]:
    manifest=prompt_manifest(); ds=load_dataset(DATASET_NAME,DATASET_CONFIG,split='train')
    groups={}
    for bucket,items in manifest['buckets'].items():
        all_items=[]
        for rec in items:
            text=ds[int(rec['row'])]['text'].strip()
            if hashlib.sha256(text.encode()).hexdigest()!=rec['text_sha256']: raise AssertionError('reserved text hash mismatch')
            enc={mid:toks[mid](text,add_special_tokens=False,return_offsets_mapping=True) for mid in MODELS}
            ref=enc[TARGET]['offset_mapping']
            if any(enc[mid]['offset_mapping']!=ref for mid in SOURCES): raise AssertionError('reserved offsets changed')
            n=len(enc[TARGET]['input_ids'])
            if n!=rec['tokens'] or not MIN_TOKENS<=n<=MAX_TOKENS: raise AssertionError('reserved token length changed')
            all_items.append({'row':int(rec['row']),'title_sha256':rec['title_sha256'],'text':text,
                'ids':{mid:enc[mid]['input_ids'] for mid in MODELS},'tokens':n})
        groups[bucket]={'map_fit':all_items[:24],'check':all_items[24:32],'test':all_items[32:48]}
    return groups

def cache_arrays(model:Any,ids:list[int]):
    with torch.inference_mode(): out=model(input_ids=torch.tensor(ids,dtype=torch.long)[None,:],use_cache=True)
    past=out.past_key_values
    if hasattr(past,'to_legacy_cache'): past=past.to_legacy_cache()
    arr=[(k[0].detach().cpu().numpy().astype(np.float32),v[0].detach().cpu().numpy().astype(np.float32)) for k,v in past]
    return arr,out

def inv_rope_np(x:np.ndarray,cfg:Any)->np.ndarray:
    z=np.asarray(x,np.float32).copy(); typ=getattr(cfg,'model_type','')
    if typ not in ('gpt_neox','llama','mistral','qwen2'): return z
    dim=int((cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct)) if typ=='gpt_neox' else z.shape[-1]
    theta=float(getattr(cfg,'rope_theta',10000.)); inv=1/(theta**(np.arange(0,dim,2,dtype=np.float64)/dim))
    ang=np.outer(np.arange(z.shape[-2],dtype=np.float64),inv); co=np.concatenate([np.cos(ang),np.cos(ang)],-1); si=-np.concatenate([np.sin(ang),np.sin(ang)],-1)
    if typ=='gpt_neox': co=np.concatenate([co,np.ones((len(co),z.shape[-1]-dim))],-1); si=np.concatenate([si,np.zeros((len(si),z.shape[-1]-dim))],-1)
    shape=(1,)*(z.ndim-2)+(len(co),co.shape[-1]); co=co.reshape(shape); si=si.reshape(shape)
    def rot(a): h=a.shape[-1]//2; return np.concatenate([-a[...,h:],a[...,:h]],-1)
    if typ=='gpt_neox':
        a,b=z[...,:dim],z[...,dim:]; return np.concatenate([a*co[...,:dim]+rot(a)*si[...,:dim],b],-1)
    return z*co+rot(z)*si

def torch_rope(x:torch.Tensor,cfg:Any)->torch.Tensor:
    typ=getattr(cfg,'model_type','')
    if typ not in ('gpt_neox','llama','mistral','qwen2'): return x
    dim=int((cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct)) if typ=='gpt_neox' else x.shape[-1]
    theta=float(getattr(cfg,'rope_theta',10000.)); idx=torch.arange(0,dim,2,dtype=x.dtype,device=x.device); inv=1/(theta**(idx/dim))
    pos=torch.arange(x.shape[0],dtype=x.dtype,device=x.device); ang=pos[:,None]*inv[None,:]
    co=torch.cat([ang.cos(),ang.cos()],-1); si=torch.cat([ang.sin(),ang.sin()],-1)
    if typ=='gpt_neox':
        co=torch.cat([co,torch.ones((len(pos),x.shape[-1]-dim),device=x.device,dtype=x.dtype)],-1); si=torch.cat([si,torch.zeros((len(pos),x.shape[-1]-dim),device=x.device,dtype=x.dtype)],-1)
    def rot(a): h=a.shape[-1]//2; return torch.cat([-a[...,h:],a[...,:h]],-1)
    if typ=='gpt_neox':
        a,b=x[...,:dim],x[...,dim:]; return torch.cat([a*co[...,:dim]+rot(a)*si[...,:dim],b],-1)
    return x*co+rot(x)*si

def nearest_layer(tl:int,src_layers:int,tgt_layers:int)->int: return int(round(tl*(src_layers-1)/max(1,tgt_layers-1)))
def source_head(h:int,src_cfg:Any,tgt_cfg:Any)->int:
    sh=getattr(src_cfg,'num_key_value_heads',None) or src_cfg.num_attention_heads
    th=getattr(tgt_cfg,'num_key_value_heads',None) or tgt_cfg.num_attention_heads
    return round(h*(sh-1)/max(1,th-1))
def role_id(layer:int,head:int,kv:int)->int: return (layer*HEADS+head)*2+kv

def role_vectors(pair:tuple[str,str],item:dict[str,Any],models:dict[str,Any])->tuple[np.ndarray,np.ndarray]:
    sm,tm=models[pair[0]],models[pair[1]]; sc,_=cache_arrays(sm,item['ids'][pair[0]]); tc,_=cache_arrays(tm,item['ids'][pair[1]])
    n=min(len(item['ids'][pair[0]]),len(item['ids'][pair[1]])); x=np.zeros((n,ROLES,D),np.float32); y=np.zeros_like(x)
    for tl in range(LAYERS):
        sl=nearest_layer(tl,sm.config.num_hidden_layers,LAYERS)
        for h in range(HEADS):
            sh=source_head(h,sm.config,tm.config)
            for kv in range(2):
                rid=role_id(tl,h,kv); xv=sc[sl][kv][sh,:n,:]; yv=tc[tl][kv][h,:n,:]
                if kv==0: xv=inv_rope_np(xv,sm.config); yv=inv_rope_np(yv,tm.config)
                x[:,rid]=xv; y[:,rid]=yv
    return x,y

def prepare(group:dict[str,list[dict[str,Any]]],split:str,models:dict[str,Any])->dict[str,list[dict[str,Any]]]:
    out={}
    for pair in PAIRS:
        out['>'.join(pair)]={}
        for part,items in group.items():
            row=[]
            for item in items:
                x,y=role_vectors(pair,item,models)
                row.append({'item':item,'x':x,'y':y})
            out['>'.join(pair)][part]=row
    return out

def sufficient(data:list[dict[str,Any]])->tuple[np.ndarray,np.ndarray]:
    sxx=np.zeros((ROLES,D+1,D+1),np.float64); sxy=np.zeros((ROLES,D+1,D),np.float64)
    for d in data:
        x=d['x'].astype(np.float64); y=d['y'].astype(np.float64); xa=np.concatenate([x,np.ones((len(x),ROLES,1))],-1)
        for r in range(ROLES): sxx[r]+=xa[:,r].T@xa[:,r]; sxy[r]+=xa[:,r].T@y[:,r]
    return sxx,sxy

def solve_maps(stats:tuple[np.ndarray,np.ndarray])->np.ndarray:
    sxx,sxy=stats; eye=np.eye(D+1); eye[-1,-1]=0.; maps=np.empty((ROLES,D+1,D),np.float32)
    for r in range(ROLES): maps[r]=np.linalg.solve(sxx[r]+ALPHA*eye,sxy[r]).astype(np.float32)
    return maps

def fit_shared_base(data_by_pair:dict[str,list[dict[str,Any]]])->tuple[np.ndarray,float]:
    t0=time.perf_counter(); stat=(np.zeros((ROLES,D+1,D+1),np.float64),np.zeros((ROLES,D+1,D),np.float64))
    for data in data_by_pair.values():
        sxx,sxy=sufficient(data); stat[0][:]+=sxx; stat[1][:]+=sxy
    return solve_maps(stat),time.perf_counter()-t0

def fit_full_maps(data:list[dict[str,Any]])->np.ndarray: return solve_maps(sufficient(data))

def map_predict(x:np.ndarray,maps:np.ndarray)->np.ndarray:
    xa=np.concatenate([x,np.ones((*x.shape[:2],1),np.float32)],-1)
    return np.einsum('nrd,rdf->nrf',xa,maps,optimize=True)

def fit_film(data:list[dict[str,Any]],base:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    zz=[]; yy=[]
    for d in data: zz.append(map_predict(d['x'],base)); yy.append(d['y'])
    z=np.concatenate(zz); y=np.concatenate(yy); gain=np.ones((ROLES,1),np.float32); bias=np.zeros((ROLES,D),np.float32)
    for r in range(ROLES):
        zr=z[:,r]; yr=y[:,r]; zm=zr.mean(0); ym=yr.mean(0); den=float(np.square(zr-zm).sum())
        if den>1e-20: gain[r,0]=float(((zr-zm)*(yr-ym)).sum()/den)
        bias[r]=ym-gain[r,0]*zm
    return gain,bias

def film_predict(x:np.ndarray,base:np.ndarray,gain:np.ndarray,bias:np.ndarray)->np.ndarray:
    return map_predict(x,base)*gain[None,:,:]+bias[None,:,:]

class RoleRepair(nn.Module):
    def __init__(self,h:int,xmean:np.ndarray,xscale:np.ndarray,outmean:np.ndarray,outscale:np.ndarray,activation:str,seed:int):
        super().__init__(); torch.manual_seed(seed); self.h=h; self.activation=activation
        self.w1=nn.Parameter(torch.randn(ROLES,D,h)*(.05/(D**.5))); self.b1=nn.Parameter(torch.zeros(ROLES,h))
        self.w2=nn.Parameter(torch.zeros(ROLES,h,D)); self.b2=nn.Parameter(torch.zeros(ROLES,D))
        self.register_buffer('xmean',torch.as_tensor(xmean,dtype=torch.float32)); self.register_buffer('xscale',torch.as_tensor(xscale,dtype=torch.float32))
        self.register_buffer('outmean',torch.as_tensor(outmean,dtype=torch.float32)); self.register_buffer('outscale',torch.as_tensor(outscale,dtype=torch.float32))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        z=((x.transpose(0,1)-self.xmean[:,None,:])/self.xscale[:,None,:])
        h=torch.einsum('rni,rih->rnh',z,self.w1)+self.b1[:,None,:]
        h=F.gelu(h) if self.activation=='gelu' else h
        o=torch.einsum('rnh,rhd->rnd',h,self.w2)+self.b2[:,None,:]
        return o.transpose(0,1)*self.outscale[None,:,:]+self.outmean[None,:,:]

def repair_normalizers(data:list[dict[str,Any]],base:np.ndarray,gain:np.ndarray|None,bias:np.ndarray|None,kind:str)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    xs=[]; outs=[]
    for d in data:
        if kind=='film': pred=film_predict(d['x'],base,gain,bias)
        elif kind=='full': pred=map_predict(d['x'],base)
        elif kind=='direct': pred=np.zeros_like(d['y'])
        else: raise ValueError(kind)
        xs.append(d['x']); outs.append(d['y']-pred)
    x=np.concatenate(xs); o=np.concatenate(outs)
    xm=x.mean(0); xsig=x.std(0).clip(1e-3); om=o.mean(0); osig=o.std(0).clip(1e-3)
    return xm.astype(np.float32),xsig.astype(np.float32),om.astype(np.float32),osig.astype(np.float32)

def maps_to_past(pred:torch.Tensor,prefix:int,cfg:Any)->tuple[tuple[torch.Tensor,torch.Tensor],...]:
    result=[]
    for l in range(LAYERS):
        ks=[]; vs=[]
        for h in range(HEADS):
            ridk=role_id(l,h,0); ridv=role_id(l,h,1)
            k=torch_rope(pred[:prefix,ridk],cfg); v=pred[:prefix,ridv]
            ks.append(k[None,None]); vs.append(v[None,None])
        result.append((torch.cat(ks,dim=1),torch.cat(vs,dim=1)))
    return tuple(result)

def train_repair(data:list[dict[str,Any]],base:np.ndarray,film:tuple[np.ndarray,np.ndarray]|None,h:int,activation:str,seed:int,models:dict[str,Any],mode:str)->tuple[RoleRepair,dict[str,Any]]:
    gain,bias=film if film is not None else (None,None)
    kind='film' if mode=='residual_film' else 'full' if mode=='residual_full' else 'direct'
    xm,xs,om,os=repair_normalizers(data,base,gain,bias,kind); net=RoleRepair(h,xm,xs,om,os,activation,seed)
    opt=torch.optim.Adam(net.parameters(),lr=LR); target=models[TARGET]; t0=time.perf_counter(); hist=[]
    ys=torch.as_tensor(np.concatenate([z['y'] for z in data]).std(0).clip(1e-3),dtype=torch.float32)
    for step in range(UPDATE_STEPS):
        d=data[(step*7)%len(data)]; x=torch.as_tensor(d['x'],dtype=torch.float32); y=torch.as_tensor(d['y'],dtype=torch.float32)
        if kind=='film': b=torch.as_tensor(film_predict(d['x'],base,gain,bias),dtype=torch.float32); pred=b+net(x)
        elif kind=='full': b=torch.as_tensor(map_predict(d['x'],base),dtype=torch.float32); pred=b+net(x)
        else: pred=net(x)
        maploss=(((pred-y)/ys[None,:,:])**2).mean()
        prefix=len(d['item']['ids'][TARGET])-2; past=maps_to_past(pred,prefix,target.config)
        q=torch.tensor([[d['item']['ids'][TARGET][-2]]],dtype=torch.long); label=torch.tensor([d['item']['ids'][TARGET][-1]],dtype=torch.long)
        out=target(input_ids=q,past_key_values=DynamicCache.from_legacy_cache(past),use_cache=True)
        ce=F.cross_entropy(out.logits[:,-1,:],label); loss=maploss+.10*ce
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
        hist.append((float(maploss.detach()),float(ce.detach())))
    return net,{'updates':UPDATE_STEPS,'seconds':time.perf_counter()-t0,'final_map_rel_mse':hist[-1][0],'final_replay_ce':hist[-1][1]}

def repair_predict(x:np.ndarray,net:RoleRepair,device='cpu')->np.ndarray:
    with torch.inference_mode(): return net(torch.as_tensor(x,dtype=torch.float32,device=device)).cpu().numpy()

def apply_method(pairkey:str,d:dict[str,Any],method:str,base:np.ndarray,full_maps:dict[str,np.ndarray],films:dict[str,tuple[np.ndarray,np.ndarray]],repairs:dict[str,dict[str,RoleRepair]],hard:list[str])->np.ndarray:
    pair=tuple(pairkey.split('>'))
    if method=='shared_linear': return map_predict(d['x'],base)
    if method=='pair_film': return film_predict(d['x'],base,*films[pairkey])
    if method=='full_ridge': return map_predict(d['x'],full_maps[pairkey])
    if method=='sparse_linear_residual':
        y=film_predict(d['x'],base,*films[pairkey])
        if pairkey in hard: y+=repair_predict(d['x'],repairs.get('sparse_linear_residual',{})[pairkey])
        return y
    if method=='sparse_mlp':
        y=film_predict(d['x'],base,*films[pairkey])
        if pairkey in hard: y+=repair_predict(d['x'],repairs.get('sparse_mlp',{})[pairkey])
        return y
    if method=='full_mlp_upper':
        y=map_predict(d['x'],full_maps[pairkey]); y+=repair_predict(d['x'],repairs.get('full_mlp_upper',{})[pairkey]); return y
    raise ValueError(method)

def nll(logits:torch.Tensor,label:int)->float: return float(F.cross_entropy(logits.reshape(1,-1),torch.tensor([label],dtype=torch.long)))
def evaluate(group_data:dict[str,list[dict[str,Any]]],base:np.ndarray,full_maps:dict[str,np.ndarray],films:dict[str,tuple[np.ndarray,np.ndarray]],repairs:dict[str,dict[str,RoleRepair]],hard:list[str],models:dict[str,Any],methods:tuple[str,...])->dict[str,Any]:
    out={'methods':{m:{'per_pair':{},'nll':[],'apply_seconds':[],'replay_seconds':[]} for m in methods},'native':{'per_pair':{},'nll':[]},'native_identity_error':[]}
    for pk,data in group_data.items():
        out['native']['per_pair'][pk]=[]
        for m in methods: out['methods'][m]['per_pair'][pk]=[]
        for d in data:
            item=d['item']; target=models[TARGET]; _,native=cache_arrays(target,item['ids'][TARGET]); label=item['ids'][TARGET][-1]; query=item['ids'][TARGET][-2]
            nnative=nll(native.logits[0,-2],label); out['native']['nll'].append(nnative); out['native']['per_pair'][pk].append(nnative)
            prefix=len(item['ids'][TARGET])-2; npast=native.past_key_values
            if hasattr(npast,'to_legacy_cache'): npast=npast.to_legacy_cache()
            native_prefix=tuple((k[:,:,:prefix,:],v[:,:,:prefix,:]) for k,v in npast)
            with torch.inference_mode(): no=target(input_ids=torch.tensor([[query]]),past_key_values=DynamicCache.from_legacy_cache(native_prefix),use_cache=True)
            out['native_identity_error'].append(abs(nnative-nll(no.logits[0,-1],label)))
            for m in methods:
                ta=time.perf_counter()
                pred=apply_method(pk,d,m,base,full_maps,films,repairs,hard)
                past=maps_to_past(torch.as_tensor(pred,dtype=torch.float32),prefix,target.config)
                out['methods'][m]['apply_seconds'].append(time.perf_counter()-ta); tr=time.perf_counter()
                with torch.inference_mode(): logits=target(input_ids=torch.tensor([[query]]),past_key_values=DynamicCache.from_legacy_cache(past),use_cache=True).logits[0,-1]
                out['methods'][m]['replay_seconds'].append(time.perf_counter()-tr)
                value=nll(logits,label); out['methods'][m]['nll'].append(value); out['methods'][m]['per_pair'][pk].append(value)
    out['native']['mean_nll']=float(np.mean(out['native']['nll']))
    for v in out['methods'].values():
        v['mean_nll']=float(np.mean(v['nll'])); v['per_pair_mean_nll']={k:float(np.mean(x)) for k,x in v['per_pair'].items()}
    out['native']['per_pair_mean_nll']={k:float(np.mean(x)) for k,x in out['native']['per_pair'].items()}
    out['native_identity_max_abs_diff']=float(max(out['native_identity_error']))
    return out

def serialize_payload(mode:str,base:np.ndarray,films:dict[str,tuple[np.ndarray,np.ndarray]],full_maps:dict[str,np.ndarray],repairs:dict[str,RoleRepair],hard:list[str])->bytes:
    meta={'mode':mode,'pairs':PAIRS,'target_layers':LAYERS,'heads':HEADS,'roles':ROLES,'hidden':{k:v.h for k,v in repairs.items()}}
    rawmeta=json.dumps(meta,separators=(',',':')).encode(); b=bytearray(struct.pack('<5sBI',b'MA883',{'shared_linear':0,'pair_film':1,'sparse_linear_residual':2,'sparse_mlp':3,'full_ridge':4,'full_mlp_upper':5}[mode],len(rawmeta))); b.extend(rawmeta)
    if mode in ('shared_linear','pair_film','sparse_linear_residual','sparse_mlp'):
        b.extend(np.asarray(base,dtype='<f4').tobytes())
    if mode in ('pair_film','sparse_linear_residual','sparse_mlp'):
        for pk in sorted(films):
            for a in films[pk]: b.extend(np.asarray(a,dtype='<f4').tobytes())
    if mode in ('full_ridge','full_mlp_upper'):
        for pk in sorted(full_maps): b.extend(np.asarray(full_maps[pk],dtype='<f4').tobytes())
    if mode in ('sparse_linear_residual','sparse_mlp','full_mlp_upper'):
        selected=sorted(repairs) if mode=='full_mlp_upper' else sorted(hard)
        selected_pairs=set(selected)
        b.extend(bytes(1 if '>'.join(pk) in selected_pairs else 0 for pk in PAIRS))
        for pk in selected:
            net=repairs[pk]
            for arr in (net.w1,net.b1,net.w2,net.b2,net.xmean,net.xscale,net.outmean,net.outscale): b.extend(np.asarray(arr.detach().cpu(),dtype='<f4').tobytes())
    return bytes(b)

def deserialize_payload(raw:bytes)->dict[str,Any]:
    magic,mode,nmeta=struct.unpack_from('<5sBI',raw,0)
    if magic!=b'MA883': raise ValueError('bad MA-883 payload')
    pos=struct.calcsize('<5sBI'); meta=json.loads(raw[pos:pos+nmeta]); pos+=nmeta
    base=np.zeros((ROLES,D+1,D),np.float32); films={}; full={}; repairs={}; hard=[]
    if mode in (0,1,2,3):
        n=ROLES*(D+1)*D; base=np.frombuffer(raw,dtype='<f4',count=n,offset=pos).reshape(ROLES,D+1,D).copy(); pos+=n*4
    if mode in (1,2,3):
        for pk in sorted('>'.join(p) for p in PAIRS):
            g=np.frombuffer(raw,dtype='<f4',count=ROLES,offset=pos).reshape(ROLES,1).copy(); pos+=ROLES*4
            b=np.frombuffer(raw,dtype='<f4',count=ROLES*D,offset=pos).reshape(ROLES,D).copy(); pos+=ROLES*D*4; films[pk]=(g,b)
    if mode in (4,5):
        for pk in sorted('>'.join(p) for p in PAIRS):
            n=ROLES*(D+1)*D; full[pk]=np.frombuffer(raw,dtype='<f4',count=n,offset=pos).reshape(ROLES,D+1,D).copy(); pos+=n*4
    if mode in (2,3,5):
        flags=np.frombuffer(raw,dtype=np.uint8,count=len(PAIRS),offset=pos); pos+=len(PAIRS)
        selected=[pk for pk,flag in zip(['>'.join(p) for p in PAIRS],flags) if flag]
        activation='identity' if mode==2 else 'gelu'
        for pk in selected:
            h=int(meta['hidden'][pk]); shapes=((ROLES,D,h),(ROLES,h),(ROLES,h,D),(ROLES,D),(ROLES,D),(ROLES,D),(ROLES,D),(ROLES,D))
            arrays=[]
            for shape in shapes:
                n=int(np.prod(shape)); arrays.append(np.frombuffer(raw,dtype='<f4',count=n,offset=pos).reshape(shape).copy()); pos+=n*4
            net=RoleRepair(h,arrays[4],arrays[5],arrays[6],arrays[7],activation,0)
            with torch.no_grad():
                for param,a in zip((net.w1,net.b1,net.w2,net.b2,net.xmean,net.xscale,net.outmean,net.outscale),arrays): param.copy_(torch.from_numpy(a))
            repairs[pk]=net; hard.append(pk)
    if pos!=len(raw): raise ValueError(f'MA-883 payload trailing bytes: {len(raw)-pos}')
    return {'mode':mode,'base':base,'films':films,'full_maps':full,'repairs':repairs,'hard':hard}

def save_shared_base(base:np.ndarray)->dict[str,Any]:
    path=Path(__file__).resolve().parents[1]/'artifacts'/'shared_base_development.bin'; path.parent.mkdir(exist_ok=True)
    raw=struct.pack('<5sHHH',b'MA883',ROLES,D+1,D)+np.asarray(base,dtype='<f4').tobytes(); path.write_bytes(raw)
    return {'path':str(path.relative_to(Path(__file__).resolve().parents[2])),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def load_shared_base()->np.ndarray:
    path=Path(__file__).resolve().parents[1]/'artifacts'/'shared_base_development.bin'; raw=path.read_bytes()
    magic,nr,di,do=struct.unpack_from('<5sHHH',raw,0)
    if (magic,nr,di,do)!=(b'MA883',ROLES,D+1,D): raise ValueError('bad frozen shared-base artifact')
    return np.frombuffer(raw,dtype='<f4',offset=struct.calcsize('<5sHHH')).reshape(ROLES,D+1,D).copy()

def model_provenance(paths:dict[str,Path])->dict[str,Any]:
    out={}
    for mid,p in paths.items():
        files=sorted(p.glob('*.safetensors')); h=hashlib.sha256()
        for f in files: h.update(f.name.encode()); h.update(f.read_bytes())
        out[mid]={'snapshot':p.name,'weight_files':len(files),'weights_sha256':h.hexdigest()}
    return out

def train_all(base:np.ndarray,data_by_pair:dict[str,list[dict[str,Any]]],films:dict[str,tuple[np.ndarray,np.ndarray]],full_maps:dict[str,np.ndarray],hard:list[str],h:int,seed:int,models:dict[str,Any],also_identity:bool=True,include_upper:bool=True)->tuple[dict[str,RoleRepair],dict[str,RoleRepair],dict[str,RoleRepair],dict[str,Any]]:
    sparse={}; linear={}; upper={}; timing={}
    for i,pk in enumerate(sorted(data_by_pair)):
        if pk in hard:
            sparse[pk],st=train_repair(data_by_pair[pk],base,films[pk],h,'gelu',seed+10+i,models,'residual_film'); timing['sparse_'+pk]=st
            if also_identity:
                linear[pk],lt=train_repair(data_by_pair[pk],base,films[pk],h,'identity',seed+10+i,models,'residual_film'); timing['linear_'+pk]=lt
        if include_upper:
            upper[pk],ut=train_repair(data_by_pair[pk],full_maps[pk],None,h,'gelu',seed+90+i,models,'residual_full'); timing['upper_'+pk]=ut
    return sparse,linear,upper,timing

def pair_key(pair)->str: return '>'.join(pair)
def id_rows(group): return {k:[x['row'] for x in v] for k,v in group.items()}

def run_dev()->dict[str,Any]:
    models,toks,paths=load_assets(); groups=load_prompt_sets(toks); dev=groups['0']; prepared=prepare(dev,'dev',models)
    fit={pk:d['map_fit'] for pk,d in prepared.items()}; shared,shared_seconds=fit_shared_base(fit); base_artifact=save_shared_base(shared); full={pk:fit_full_maps(v) for pk,v in fit.items()}
    films={pk:fit_film(data,shared) for pk,data in fit.items()}
    check={pk:d['check'] for pk,d in prepared.items()}; check_eval=evaluate(check,shared,full,films,{},[],models,('shared_linear','pair_film','full_ridge'))
    hard=[pk for pk in sorted(fit) if check_eval['methods']['pair_film']['per_pair_mean_nll'][pk]-check_eval['native']['per_pair_mean_nll'][pk]>1.0]
    width_rows={}; t0=time.perf_counter()
    selected_width=HIDDEN_CANDIDATES[0]
    if hard:
        for h in HIDDEN_CANDIDATES:
            sparse,linear,upper,timing=train_all(shared,fit,films,full,hard,h,DEV_BUCKET*100+h,models,also_identity=True,include_upper=False)
            test={pk:d['test'] for pk,d in prepared.items()}
            ev=evaluate(test,shared,full,films,{'sparse_mlp':sparse},hard,models,('sparse_mlp',))
            width_rows[str(h)]={'sparse_mean_nll':ev['methods']['sparse_mlp']['mean_nll'],'sparse_by_pair':ev['methods']['sparse_mlp']['per_pair_mean_nll'],'training':timing}
        selected_width=min(HIDDEN_CANDIDATES,key=lambda h:(np.mean([width_rows[str(h)]['sparse_by_pair'][pk] for pk in hard]),h))
    sparse,linear,upper,timing=train_all(shared,fit,films,full,hard,selected_width,DEV_BUCKET*1000+selected_width,models,also_identity=True,include_upper=True)
    test={pk:d['test'] for pk,d in prepared.items()}; ev=evaluate(test,shared,full,films,{'sparse_linear_residual':linear,'sparse_mlp':sparse},hard,models,('shared_linear','pair_film','full_ridge','sparse_linear_residual','sparse_mlp'))
    if upper:
        # Evaluate upper controls separately because upper residual networks are stored independently.
        upev=evaluate(test,shared,full,films,{'full_mlp_upper':upper},list(upper),models,('full_mlp_upper',))
    else: upev=None
    return {'experiment_id':'MA-883','development_bucket':0,'hard_pair_keys':hard,'selected_hidden_width':selected_width,'hard_pair_fraction':len(hard)/len(PAIRS),'shared_base_artifact':base_artifact,
      'width_selection':width_rows,'shared_fit_seconds':shared_seconds,'final_repair_training':timing,'check_metrics':check_eval,'test_metrics':ev,'upper_test_metrics':upev,
      'prompt_rows':{pk:id_rows(dev['map_fit']) for pk in []}|{'map_fit':[[x['row'] for x in dev['map_fit']]],'check':[[x['row'] for x in dev['check']]],'test':[[x['row'] for x in dev['test']]]},
      'models':model_provenance(paths),'runtime':{'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'dev_seconds':time.perf_counter()-t0}}

def run_fresh(seeds:tuple[int,...],hard:list[str],h:int)->dict[str,Any]:
    models,toks,paths=load_assets(); groups=load_prompt_sets(toks); base=load_shared_base(); rows=[]
    for seed in seeds:
        bucket=str(FRESH_SEEDS.index(seed)+1); prepared=prepare(groups[bucket],'fresh',models)
        fit={pk:d['map_fit'] for pk,d in prepared.items()}; t0=time.perf_counter()
        full={pk:fit_full_maps(data) for pk,data in fit.items()}; films={pk:fit_film(data,base) for pk,data in fit.items()}
        sparse,linear,upper,training=train_all(base,fit,films,full,hard,h,seed,models,also_identity=True,include_upper=True)
        check={pk:d['check'] for pk,d in prepared.items()}; test={pk:d['test'] for pk,d in prepared.items()}
        repairs={'sparse_linear_residual':linear,'sparse_mlp':sparse,'full_mlp_upper':upper}
        methods=('shared_linear','pair_film','full_ridge','sparse_linear_residual','sparse_mlp','full_mlp_upper')
        check_ev=evaluate(check,base,full,films,repairs,hard,models,methods)
        ev=evaluate(test,base,full,films,repairs,hard,models,methods)
        variants={
          'shared_linear':serialize_payload('shared_linear',base,films,full,{},[]),
          'pair_film':serialize_payload('pair_film',base,films,full,{},[]),
          'full_ridge':serialize_payload('full_ridge',base,films,full,{},[]),
          'sparse_linear_residual':serialize_payload('sparse_linear_residual',base,films,full,linear,hard),
          'sparse_mlp':serialize_payload('sparse_mlp',base,films,full,sparse,hard),
          'full_mlp_upper':serialize_payload('full_mlp_upper',base,films,full,upper,sorted(upper))}
        artifact=Path(__file__).resolve().parents[1]/'artifacts'; artifact.mkdir(exist_ok=True); outpaths={}; hashes={}
        for name,raw in variants.items():
            out=artifact/f'{name}_seed_{seed}.bin'; out.write_bytes(raw); outpaths[name]=str(out.relative_to(Path(__file__).resolve().parents[2])); hashes[name]=hashlib.sha256(raw).hexdigest()
        full_bytes=len(variants['full_ridge']); pairfilm_bytes=len(variants['pair_film']); candidate_bytes=len(variants['sparse_mlp'])
        private_bytes=candidate_bytes-pairfilm_bytes
        token_counts={pk:sum(x['item']['tokens'] for x in data) for pk,data in test.items()}
        repaired=sum(token_counts[pk] for pk in hard); token_total=sum(token_counts.values())
        row={'seed':seed,'article_bucket':int(bucket),'prompt_rows':{pk:{k:[z['item']['row'] for z in v] for k,v in split.items()} for pk,split in prepared.items()},
             'check_metrics':check_ev,'test_metrics':ev,'hard_pair_keys':hard,'repaired_pair_fraction':len(hard)/len(PAIRS),'repaired_token_fraction':repaired/max(1,token_total),
             'payload_bytes':{k:len(v) for k,v in variants.items()},'private_repair_bytes':private_bytes,'private_fraction_of_full_ridge':private_bytes/full_bytes,
             'total_sparse_fraction_of_full_ridge':candidate_bytes/full_bytes,'payload_sha256':hashes,'artifact_paths':outpaths,'training':training,
             'compute':{'fresh_fit_repair_eval_seconds':time.perf_counter()-t0,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
        rows.append(row)
        partial={'experiment_id':'MA-883','hard_pair_keys':hard,'selected_hidden_width':h,'seeds_completed':[x['seed'] for x in rows],'rows':rows,'models':model_provenance(paths)}
        (Path(__file__).resolve().parents[1]/'fresh_results.partial.json').write_text(json.dumps(partial,indent=2)+'\n')
    return {'experiment_id':'MA-883','seeds':list(seeds),'hard_pair_keys':hard,'selected_hidden_width':h,'rows':rows,'models':model_provenance(paths),
            'runtime':{'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['dev','fresh']); ap.add_argument('--hard',default=''); ap.add_argument('--width',type=int,default=4); ap.add_argument('--seeds',default='983,997,1009')
    args=ap.parse_args()
    if args.mode=='dev': result=run_dev(); name='development_results.json'
    else: result=run_fresh(tuple(int(x) for x in args.seeds.split(',')),[x for x in args.hard.split(';') if x],args.width); name='fresh_results.json'
    out=Path(__file__).resolve().parents[1]/name; out.write_text(json.dumps(result,indent=2)+'\n'); print(out)
