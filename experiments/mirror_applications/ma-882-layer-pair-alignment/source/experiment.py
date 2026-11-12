"""MA-882 fixed-route cross-layer KV translator screen."""
from __future__ import annotations
import os
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import hashlib, json, resource, struct, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

SOURCES=('EleutherAI/pythia-70m-deduped','EleutherAI/pythia-160m-deduped')
TARGET='HuggingFaceTB/SmolLM2-135M'
PAIRS=tuple((x,TARGET) for x in SOURCES)
MODELS=SOURCES+(TARGET,)
ANCHOR_REL=(0.0,0.5,1.0); PCA_RANKS=(32,64,128,256); CP_RANKS=(2,4,8,16,32)
FRESH_SEEDS=(983,997,1009); DEV_SEED=829; D=64; ALPHA=.01
MIN_TOKENS=16; MAX_TOKENS=128; NROLES=6; TARGET_LAYERS=30; TARGET_HEADS=3

def snap(mid:str)->Path:
    p=Path.home()/'.cache/huggingface/hub'/('models--'+mid.replace('/','--'))/'snapshots'
    if not p.exists(): raise FileNotFoundError(f'cached model missing: {mid}')
    return sorted(p.iterdir())[0]

def load_assets():
    torch.set_num_threads(min(4,os.cpu_count() or 1)); models={}; toks={}; paths={}
    for mid in MODELS:
        path=snap(mid); paths[mid]=path
        toks[mid]=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True)
        models[mid]=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,attn_implementation='eager').eval()
    expected=(models[TARGET].config.num_hidden_layers,getattr(models[TARGET].config,'num_key_value_heads',None) or models[TARGET].config.num_attention_heads)
    if expected!=(TARGET_LAYERS,TARGET_HEADS): raise ValueError(f'unexpected target layer/head grid: {expected}')
    return models,toks,paths

def dataset(split:str): return load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split,download_mode='reuse_dataset_if_exists')

def eligible(split:str,pair:tuple[str,str],toks:dict[str,Any],minimum:int)->list[dict[str,Any]]:
    out=[]; seen=set(); st,tt=toks[pair[0]],toks[pair[1]]
    for i,row in enumerate(dataset(split)):
        text=row['text'].strip()
        if len(text)<32 or text in seen: continue
        a=st(text,add_special_tokens=False,return_offsets_mapping=True); b=tt(text,add_special_tokens=False,return_offsets_mapping=True)
        if a['offset_mapping']!=b['offset_mapping']: continue
        n=len(a['input_ids'])
        if MIN_TOKENS<=n<=MAX_TOKENS:
            seen.add(text); out.append({'row':i,'text':text,'src_ids':a['input_ids'],'tgt_ids':b['input_ids'],'tokens':n})
            if len(out)>=minimum: break
    if len(out)<minimum: raise RuntimeError(f'{split} {pair}: need {minimum} eligible prompts, found {len(out)}')
    return out

def prompt_chunks(split:str,pair:tuple[str,str],toks:dict[str,Any],seed:int,role:str)->list[dict[str,Any]]:
    if split=='train':
        arr=eligible(split,pair,toks,48); tag=sum((i+1)*ord(c) for i,c in enumerate('>'.join(pair)))
        perm=np.random.default_rng(DEV_SEED+tag).permutation(len(arr)); chosen=[arr[int(i)] for i in perm]
        ranges={'map_fit':(0,24),'route':(24,32),'test':(32,48)}
        a,b=ranges[role]; return chosen[a:b]
    count=48 if role in ('map_fit','route','calibration') else 36
    arr=eligible(split,pair,toks,count); tag=sum((i+1)*ord(c) for i,c in enumerate('>'.join(pair)))
    salt=0 if split=='validation' else 7919
    perm=np.random.default_rng(51236+tag+salt).permutation(len(arr))
    group=FRESH_SEEDS.index(seed)
    if role=='calibration': a,b=group*16,(group+1)*16
    elif role=='map_fit': a,b=group*16,group*16+12
    elif role=='route': a,b=group*16+12,(group+1)*16
    elif role=='test': a,b=group*12,(group+1)*12
    else: raise ValueError(role)
    return [arr[int(i)] for i in perm[a:b]]

def cache(model:Any,ids:list[int]):
    with torch.inference_mode(): out=model(input_ids=torch.tensor(ids,dtype=torch.long)[None,:],use_cache=True)
    past=out.past_key_values
    if hasattr(past,'to_legacy_cache'): past=past.to_legacy_cache()
    arrays=[(k[0].detach().cpu().numpy().astype(np.float64),v[0].detach().cpu().numpy().astype(np.float64)) for k,v in past]
    return arrays,out

def rotate_half(x:np.ndarray)->np.ndarray:
    h=x.shape[-1]//2; return np.concatenate([-x[...,h:],x[...,:h]],axis=-1)

def rope(x:np.ndarray,cfg:Any,inverse:bool=False)->np.ndarray:
    z=np.asarray(x,dtype=np.float64).copy(); typ=getattr(cfg,'model_type','')
    if typ not in ('gpt_neox','llama','mistral','qwen2'): return z
    dim=(cfg.hidden_size//cfg.num_attention_heads)*float(cfg.rotary_pct) if typ=='gpt_neox' else z.shape[-1]
    dim=int(dim); theta=float(getattr(cfg,'rope_theta',10000.)); inv=1/(theta**(np.arange(0,dim,2,dtype=np.float64)/dim))
    ang=np.outer(np.arange(z.shape[-2],dtype=np.float64),inv); co=np.concatenate([np.cos(ang),np.cos(ang)],-1); si=np.concatenate([np.sin(ang),np.sin(ang)],-1)
    if typ=='gpt_neox':
        co=np.concatenate([co,np.ones((len(co),z.shape[-1]-dim))],-1); si=np.concatenate([si,np.zeros((len(si),z.shape[-1]-dim))],-1)
    shape=(1,)*(z.ndim-2)+(len(co),co.shape[-1]); co=co.reshape(shape); si=si.reshape(shape)
    if inverse: si=-si
    if typ=='gpt_neox':
        a,b=z[...,:dim],z[...,dim:]; return np.concatenate([a*co[...,:dim]+rotate_half(a)*si[...,:dim],b],-1)
    return z*co+rotate_half(z)*si

def source_anchors(config:Any)->tuple[int,...]:
    n=config.num_hidden_layers
    return tuple(int(round(x*(n-1))) for x in ANCHOR_REL)

def source_head(target_head:int,source_config:Any,target_config:Any)->int:
    sh=getattr(source_config,'num_key_value_heads',None) or source_config.num_attention_heads
    th=getattr(target_config,'num_key_value_heads',None) or target_config.num_attention_heads
    return round(target_head*(sh-1)/max(1,th-1))

def fit_maps(pair:tuple[str,str],prompts:list[dict[str,Any]],models:dict[str,Any])->dict[str,Any]:
    sm,tm=models[pair[0]],models[pair[1]]; anchors=source_anchors(sm.config); layers=tm.config.num_hidden_layers
    hkv=getattr(tm.config,'num_key_value_heads',None) or tm.config.num_attention_heads
    sxx=np.zeros((3,layers,hkv,2,D+1,D+1),np.float64)
    sxy=np.zeros((3,layers,hkv,2,D+1,D),np.float64)
    t0=time.perf_counter(); token_count=0
    for item in prompts:
        sc,_=cache(sm,item['src_ids']); tc,_=cache(tm,item['tgt_ids']); n=min(len(item['src_ids']),len(item['tgt_ids'])); token_count+=n
        for a,sl in enumerate(anchors):
            for tl in range(layers):
                for h in range(hkv):
                    sh=source_head(h,sm.config,tm.config)
                    for kv in range(2):
                        x=sc[sl][kv][sh,:n,:]; y=tc[tl][kv][h,:n,:]
                        if kv==0: x=rope(x,sm.config,True); y=rope(y,tm.config,True)
                        xa=np.concatenate([x,np.ones((n,1),np.float64)],axis=1)
                        sxx[a,tl,h,kv]+=xa.T@xa; sxy[a,tl,h,kv]+=xa.T@y
    maps=np.empty((3,layers,hkv,2,D+1,D),np.float32); eye=np.eye(D+1); eye[-1,-1]=0.
    for a in range(3):
        for tl in range(layers):
            for h in range(hkv):
                for kv in range(2):
                    maps[a,tl,h,kv]=np.linalg.solve(sxx[a,tl,h,kv]+ALPHA*eye,sxy[a,tl,h,kv]).astype(np.float32)
    return {'pair':pair,'anchors':anchors,'maps':maps,'fit_seconds':time.perf_counter()-t0,'tokens':token_count}

def choose_routes(mapset:dict[str,Any],prompts:list[dict[str,Any]],models:dict[str,Any])->np.ndarray:
    pair=mapset['pair']; sm,tm=models[pair[0]],models[pair[1]]; scores=np.zeros((3,TARGET_LAYERS),np.float64); counts=np.zeros_like(scores)
    for item in prompts:
        sc,_=cache(sm,item['src_ids']); tc,_=cache(tm,item['tgt_ids']); n=min(len(item['src_ids']),len(item['tgt_ids']))
        for tl in range(TARGET_LAYERS):
            for a,sl in enumerate(mapset['anchors']):
                num=den=0.
                for h in range(TARGET_HEADS):
                    sh=source_head(h,sm.config,tm.config)
                    for kv in range(2):
                        x=sc[sl][kv][sh,:n,:]; y=tc[tl][kv][h,:n,:]
                        if kv==0: x=rope(x,sm.config,True); y=rope(y,tm.config,True)
                        xa=np.concatenate([x,np.ones((n,1),np.float64)],axis=1)
                        pred=xa@mapset['maps'][a,tl,h,kv]
                        num+=float(np.square(pred-y).sum()); den+=float(np.square(y).sum())
                scores[a,tl]+=num/max(den,1e-30); counts[a,tl]+=1
    avg=scores/np.maximum(counts,1.)
    routes=np.argsort(avg,axis=0,kind='stable')[:2].T.astype(np.int8)
    return routes

def pack_selected(mapsets:list[dict[str,Any]],routes:list[np.ndarray])->tuple[np.ndarray,np.ndarray,list[tuple[int,int,int,int]]]:
    bank=np.zeros((len(mapsets),3,TARGET_LAYERS,TARGET_HEADS,2,D+1,D),np.float32); mask=np.zeros((len(mapsets),3,TARGET_LAYERS,6),bool); ids=[]
    for p,(m,route) in enumerate(zip(mapsets,routes)):
        for tl in range(TARGET_LAYERS):
            for a in route[tl]:
                for h in range(TARGET_HEADS):
                    for kv in range(2):
                        role=2*h+kv; bank[p,a,tl,h,kv]=m['maps'][a,tl,h,kv]; mask[p,a,tl,role]=True; ids.append((p,int(a),tl,role))
    return bank,mask,ids

def flatten_map(bank:np.ndarray,p:int,a:int,t:int,role:int)->np.ndarray:
    return bank[p,a,t,role//2,role%2].reshape(-1).astype(np.float64)

def fit_basis(bank:np.ndarray,ids:list[tuple[int,int,int,int]],max_rank:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    x=np.stack([flatten_map(bank,*idx) for idx in ids]); mean=x.mean(0); _,_,vt=np.linalg.svd(x-mean,full_matrices=False)
    return mean.astype(np.float32),vt[:max_rank].astype(np.float32),x.astype(np.float32)

def basis_codes(bank:np.ndarray,ids:list[tuple[int,int,int,int]],mean:np.ndarray,basis:np.ndarray,rank:int)->np.ndarray:
    vals=np.stack([flatten_map(bank,*idx) for idx in ids]).astype(np.float32)
    return (vals-mean)@basis[:rank].T

def decode_pca(ids:list[tuple[int,int,int,int]],codes:np.ndarray,mean:np.ndarray,basis:np.ndarray,rank:int)->np.ndarray:
    bank=np.zeros((2,3,TARGET_LAYERS,TARGET_HEADS,2,D+1,D),np.float32)
    for idx,code in zip(ids,codes):
        v=(mean+code@basis[:rank]).reshape(D+1,D); p,a,t,role=idx; bank[p,a,t,role//2,role%2]=v
    return bank

def tensorize_codes(ids:list[tuple[int,int,int,int]],codes:np.ndarray,rank:int)->tuple[np.ndarray,np.ndarray]:
    tensor=np.zeros((2,3,TARGET_LAYERS,6,rank),np.float32); mask=np.zeros((2,3,TARGET_LAYERS,6),bool)
    for idx,code in zip(ids,codes): tensor[idx]=code; mask[idx]=True
    return tensor,mask

def cp_reconstruct(factors:list[torch.Tensor])->torch.Tensor:
    return torch.einsum('aq,bq,cq,dq,eq->abcde',*factors)

def fit_cp(tensor:np.ndarray,mask:np.ndarray,q:int,seed:int,steps:int=300)->tuple[list[np.ndarray],dict[str,Any]]:
    torch.manual_seed(seed); x=torch.as_tensor(tensor); m=torch.as_tensor(mask)[...,None].expand_as(x); factors=[torch.nn.Parameter(torch.randn((n,q))*0.05) for n in (*tensor.shape[:-1],tensor.shape[-1])]
    opt=torch.optim.Adam(factors,lr=.03); t0=time.perf_counter(); lossval=float('nan')
    for _ in range(steps):
        pred=cp_reconstruct(factors); loss=(pred-x).square()[m].mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(factors,10.); opt.step(); lossval=float(loss.detach())
    return [v.detach().numpy() for v in factors],{'updates':steps,'seconds':time.perf_counter()-t0,'masked_mse':lossval}

def decode_cp(ids:list[tuple[int,int,int,int]],factors:list[np.ndarray],mean:np.ndarray,basis:np.ndarray,rank:int)->np.ndarray:
    tensor=cp_reconstruct([torch.from_numpy(x) for x in factors]).detach().numpy(); codes=np.stack([tensor[idx] for idx in ids])
    return decode_pca(ids,codes,mean,basis,rank)

def nearest_routes(mapsets:list[dict[str,Any]])->list[np.ndarray]:
    out=[]
    for m in mapsets:
        n=len(m['anchors']); nl=TARGET_LAYERS; ns=models_layers(m)
        route=[]
        for tl in range(nl):
            rel=tl/max(1,nl-1); route.append([int(np.argmin(np.abs(np.asarray(ANCHOR_REL)-rel)))])
        out.append(np.asarray(route,dtype=np.int8))
    return out

def models_layers(mapset:dict[str,Any])->int: return mapset['maps'].shape[1]

def apply_bank(source_cache:list[tuple[np.ndarray,np.ndarray]],bank:np.ndarray,pair_index:int,routes:np.ndarray,src_config:Any,tgt_config:Any,prefix_len:int)->tuple[tuple[torch.Tensor,torch.Tensor],...]:
    layers=tgt_config.num_hidden_layers; hkv=getattr(tgt_config,'num_key_value_heads',None) or tgt_config.num_attention_heads
    outk=np.zeros((layers,hkv,prefix_len,D),np.float32); outv=np.zeros_like(outk)
    for tl in range(layers):
        selected=routes[tl]
        for h in range(hkv):
            sh=source_head(h,src_config,tgt_config)
            for kv in range(2):
                ys=[]
                for a in selected:
                    sl=source_anchors(src_config)[int(a)]; x=source_cache[sl][kv][sh,:prefix_len,:]
                    if kv==0: x=rope(x,src_config,True)
                    xa=np.concatenate([x,np.ones((prefix_len,1),np.float64)],axis=1)
                    y=(xa@bank[pair_index,int(a),tl,h,kv]).astype(np.float32)
                    if kv==0: y=rope(y,tgt_config).astype(np.float32)
                    ys.append(y)
                y=np.mean(ys,axis=0)
                (outk if kv==0 else outv)[tl,h]=y
    return tuple((torch.from_numpy(outk[l:l+1]),torch.from_numpy(outv[l:l+1])) for l in range(layers))

def native_legacy(out:Any,prefix_len:int):
    past=out.past_key_values
    if hasattr(past,'to_legacy_cache'): past=past.to_legacy_cache()
    return tuple((k[:,:,:prefix_len,:].contiguous(),v[:,:,:prefix_len,:].contiguous()) for k,v in past)

def nll(logits:torch.Tensor,label:int)->float: return float(F.cross_entropy(logits.reshape(1,-1),torch.tensor([label],dtype=torch.long)))

def evaluate(method_banks:dict[str,np.ndarray],routes:list[np.ndarray],prompts_by_pair:list[list[dict[str,Any]]],mapsets:list[dict[str,Any]],models:dict[str,Any])->dict[str,Any]:
    result={'methods':{k:{'nll':[],'apply_seconds':[],'replay_seconds':[]} for k in method_banks},'native_nll':[],'native_replay_nll':[],'identity_absdiff':[],'prompt_rows':[]}
    target=models[TARGET]
    for pi,pair in enumerate(PAIRS):
        src=models[pair[0]]; prompt_list=prompts_by_pair[pi]
        for item in prompt_list:
            sc,_=cache(src,item['src_ids']); tc,to=cache(target,item['tgt_ids']); result['prompt_rows'].append({'pair':list(pair),'row':item['row']})
            label=item['tgt_ids'][-1]; query=item['tgt_ids'][-2]; prefix_len=len(item['tgt_ids'])-2
            native_value=nll(to.logits[0,-2],label); result['native_nll'].append(native_value)
            npast=native_legacy(to,prefix_len)
            with torch.inference_mode(): native_out=target(input_ids=torch.tensor([[query]]),past_key_values=DynamicCache.from_legacy_cache(npast),use_cache=True)
            nv=nll(native_out.logits[0,-1],label); result['native_replay_nll'].append(nv); result['identity_absdiff'].append(abs(native_value-nv))
            for name,bank in method_banks.items():
                t0=time.perf_counter(); past=apply_bank(sc,bank,pi,routes[pi],src.config,target.config,prefix_len); result['methods'][name]['apply_seconds'].append(time.perf_counter()-t0)
                t1=time.perf_counter()
                with torch.inference_mode(): out=target(input_ids=torch.tensor([[query]]),past_key_values=DynamicCache.from_legacy_cache(past),use_cache=True)
                result['methods'][name]['replay_seconds'].append(time.perf_counter()-t1)
                result['methods'][name]['nll'].append(nll(out.logits[0,-1],label))
    for m in result['methods'].values(): m['mean_nll']=float(np.mean(m['nll']))
    result['native_mean_nll']=float(np.mean(result['native_nll'])); result['native_replay_mean_nll']=float(np.mean(result['native_replay_nll']))
    result['identity_max_abs_nll_diff']=float(max(result['identity_absdiff']))
    return result

def route_bytes(routes:list[np.ndarray])->bytes: return np.stack(routes).astype(np.uint8).tobytes()
def serialize_full(bank:np.ndarray,ids:list[tuple[int,int,int,int]],routes:list[np.ndarray])->bytes:
    meta=json.dumps({'pairs':PAIRS,'target_layers':TARGET_LAYERS,'target_heads':TARGET_HEADS,'roles':NROLES,'map_shape':[D+1,D]},separators=(',',':')).encode()
    vals=np.stack([bank[idx[0],idx[1],idx[2],idx[3]//2,idx[3]%2] for idx in ids])
    return struct.pack('<5sBI',b'MA882',0,len(meta))+meta+route_bytes(routes)+np.asarray(vals,dtype='<f4').tobytes()
def serialize_pca(mean:np.ndarray,basis:np.ndarray,codes:np.ndarray,rank:int,ids:list[tuple[int,int,int,int]],routes:list[np.ndarray])->bytes:
    meta=json.dumps({'pairs':PAIRS,'target_layers':TARGET_LAYERS,'roles':NROLES,'rank':rank},separators=(',',':')).encode()
    return struct.pack('<5sBI',b'MA882',1,len(meta))+meta+route_bytes(routes)+np.asarray(mean,dtype='<f4').tobytes()+np.asarray(basis[:rank],dtype='<f4').tobytes()+np.asarray(codes,dtype='<f4').tobytes()
def serialize_cp(mean:np.ndarray,basis:np.ndarray,factors:list[np.ndarray],rank:int,q:int,ids:list[tuple[int,int,int,int]],routes:list[np.ndarray])->bytes:
    meta=json.dumps({'pairs':PAIRS,'target_layers':TARGET_LAYERS,'roles':NROLES,'rank':rank,'cp_rank':q},separators=(',',':')).encode()
    b=bytearray(struct.pack('<5sBI',b'MA882',2,len(meta))); b.extend(meta); b.extend(route_bytes(routes)); b.extend(np.asarray(mean,dtype='<f4').tobytes()); b.extend(np.asarray(basis[:rank],dtype='<f4').tobytes())
    for a in factors: b.extend(np.asarray(a,dtype='<f4').tobytes())
    return bytes(b)

def decode_payload(raw:bytes)->tuple[str,np.ndarray,list[np.ndarray]]:
    magic,mode,nmeta=struct.unpack_from('<5sBI',raw,0)
    if magic!=b'MA882': raise ValueError('bad MA-882 payload header')
    pos=struct.calcsize('<5sBI'); meta=json.loads(raw[pos:pos+nmeta]); pos+=nmeta
    nroutes=2*TARGET_LAYERS*2
    routes=np.frombuffer(raw,dtype=np.uint8,count=nroutes,offset=pos).reshape(2,TARGET_LAYERS,2).astype(np.int8); pos+=nroutes
    ids=[(p,int(a),t,role) for p in range(2) for t in range(TARGET_LAYERS) for a in routes[p,t]
         for role in range(NROLES)]
    bank=np.zeros((2,3,TARGET_LAYERS,TARGET_HEADS,2,D+1,D),np.float32)
    if mode==0:
        vals=np.frombuffer(raw,dtype='<f4',count=len(ids)*(D+1)*D,offset=pos).reshape(len(ids),D+1,D).copy()
        pos+=len(ids)*(D+1)*D*4
        for idx,val in zip(ids,vals):
            p,a,t,role=idx; bank[p,a,t,role//2,role%2]=val
    elif mode in (1,2):
        rank=int(meta['rank']); size=(D+1)*D
        mean=np.frombuffer(raw,dtype='<f4',count=size,offset=pos).copy(); pos+=size*4
        basis=np.frombuffer(raw,dtype='<f4',count=rank*size,offset=pos).reshape(rank,size).copy(); pos+=rank*size*4
        if mode==1:
            codes=np.frombuffer(raw,dtype='<f4',count=len(ids)*rank,offset=pos).reshape(len(ids),rank).copy()
            pos+=len(ids)*rank*4
        else:
            q=int(meta['cp_rank']); dims=((2,q),(3,q),(TARGET_LAYERS,q),(NROLES,q),(rank,q)); factors=[]
            for shape in dims:
                n=int(np.prod(shape)); factors.append(np.frombuffer(raw,dtype='<f4',count=n,offset=pos).reshape(shape).copy()); pos+=n*4
            tensor=cp_reconstruct([torch.from_numpy(x) for x in factors]).detach().numpy()
            codes=np.stack([tensor[idx] for idx in ids])
        vals=mean[None,:]+codes@basis
        for idx,val in zip(ids,vals):
            p,a,t,role=idx; bank[p,a,t,role//2,role%2]=val.reshape(D+1,D)
    else: raise ValueError(f'unknown MA-882 payload mode {mode}')
    if pos!=len(raw): raise ValueError(f'payload has {len(raw)-pos} trailing bytes')
    return ('full' if mode==0 else 'pca' if mode==1 else 'cp'),bank,[routes[p] for p in range(2)]

def provenance(paths:dict[str,Path])->dict[str,Any]:
    out={}
    for mid,p in paths.items():
        files=list(p.glob('*.safetensors')); h=hashlib.sha256()
        for f in files: h.update(f.name.encode()); h.update(f.read_bytes())
        out[mid]={'snapshot':p.name,'weight_files':len(files),'weights_sha256':h.hexdigest()}
    return out

def aggregate_nll(ev:dict[str,Any],method:str)->float:
    return ev['native_mean_nll'] if method=='native' else ev['methods'][method]['mean_nll']

def make_variants(mapsets:list[dict[str,Any]],routes:list[np.ndarray],rank:int,cp_q:int|None,cp_seed:int)->dict[str,Any]:
    full,mask,ids=pack_selected(mapsets,routes); mean,basis,_=fit_basis(full,ids,rank); codes=basis_codes(full,ids,mean,basis,rank); pca=decode_pca(ids,codes,mean,basis,rank)
    variants={'full':full,'pca':pca}; serial={'full':serialize_full(full,ids,routes),'pca':serialize_pca(mean,basis,codes,rank,ids,routes)}
    fitstats={}
    if cp_q is not None:
        tensor,cmask=tensorize_codes(ids,codes,rank); factors,fitstats=fit_cp(tensor,cmask,cp_q,cp_seed); cpbank=decode_cp(ids,factors,mean,basis,rank)
        variants['cp']=cpbank; serial['cp']=serialize_cp(mean,basis,factors,rank,cp_q,ids,routes)
    return {'variants':variants,'serial':serial,'mean':mean,'basis':basis,'codes':codes,'ids':ids,'routes':routes,'cp_fit':fitstats,'mask':mask}

def pair_manifest(split:str,seed:int|None,toks:dict[str,Any],role:str)->list[list[dict[str,Any]]]:
    return [prompt_chunks(split,p,toks,seed or DEV_SEED,role) for p in PAIRS]

def run_dev()->dict[str,Any]:
    models,toks,paths=load_assets(); cal=pair_manifest('train',None,toks,'map_fit'); sel=pair_manifest('train',None,toks,'route'); test=pair_manifest('train',None,toks,'test')
    mapsets=[]; routes=[]; t0=time.perf_counter()
    for i,p in enumerate(PAIRS):
        m=fit_maps(p,cal[i],models); mapsets.append(m); routes.append(choose_routes(m,sel[i],models))
    base=make_variants(mapsets,routes,max(PCA_RANKS),None,DEV_SEED); evfull=evaluate({'full':base['variants']['full'],'nearest':base['variants']['full']},routes,test,mapsets,models)
    nearest=nearest_routes(mapsets); nearest_bank,_,_=pack_selected(mapsets,nearest); evnearest=evaluate({'nearest':nearest_bank},nearest,test,mapsets,models)
    full_nll=aggregate_nll(evfull,'full'); native=evfull['native_mean_nll']; rows={}; pca_stats=[]
    for rank in PCA_RANKS:
        v=make_variants(mapsets,routes,rank,None,DEV_SEED+rank); ev=evaluate({'pca':v['variants']['pca']},routes,test,mapsets,models)
        serial=v['serial']['pca']; nllv=aggregate_nll(ev,'pca'); rows[str(rank)]={'mean_nll':nllv,'delta_full':nllv-full_nll,'delta_native':nllv-native,'payload_bytes':len(serial),'evaluation':ev}
        pca_stats.append((rank,nllv,len(serial)))
    qualifies=[x for x in pca_stats if x[1]<=full_nll+.5 and x[1]<=native+1.]
    rank=min(qualifies,key=lambda x:x[2])[0] if qualifies else min(pca_stats,key=lambda x:x[1])[0]
    pv=make_variants(mapsets,routes,rank,None,DEV_SEED); pe=evaluate({'pca':pv['variants']['pca']},routes,test,mapsets,models); pca_nll=aggregate_nll(pe,'pca')
    cp_rows={}
    for q in CP_RANKS:
        v=make_variants(mapsets,routes,rank,q,DEV_SEED+rank*100+q); ev=evaluate({'cp':v['variants']['cp']},routes,test,mapsets,models)
        nn=aggregate_nll(ev,'cp'); cp_rows[str(q)]={'mean_nll':nn,'delta_full':nn-full_nll,'delta_pca':nn-pca_nll,'delta_native':nn-native,'payload_bytes':len(v['serial']['cp']),'fit':v['cp_fit']}
    cqual=[(q,d['mean_nll'],d['payload_bytes']) for q,d in cp_rows.items() if d['mean_nll']<=full_nll+.5 and d['mean_nll']<=pca_nll+.25 and d['mean_nll']<=native+1.]
    cq=min(cqual,key=lambda x:x[2])[0] if cqual else min(((int(q),d['mean_nll'],d['payload_bytes']) for q,d in cp_rows.items()),key=lambda x:x[1])[0]
    return {'experiment_id':'MA-882','selected_pca_rank':rank,'selected_cp_rank':int(cq),'pca_selection_rule_met':bool(qualifies),'cp_selection_rule_met':bool(cqual),
            'full_mean_nll':full_nll,'native_mean_nll':native,'nearest_depth_mean_nll':aggregate_nll(evnearest,'nearest'),'selected_pca_mean_nll':pca_nll,
            'pca_rank_results':rows,'cp_rank_results':cp_rows,'prompt_rows':{'map_fit':[[x['row'] for x in z] for z in cal],'route':[[x['row'] for x in z] for z in sel],'test':[[x['row'] for x in z] for z in test]},
            'routes':[[x.tolist() for x in r] for r in routes],'fresh_seeds':list(FRESH_SEEDS),'models':provenance(paths),
            'native_identity_max_abs_nll_diff':evfull['identity_max_abs_nll_diff'],'runtime':{'dev_seconds':time.perf_counter()-t0,'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}

def run_fresh(seeds:tuple[int,...],rank:int,q:int)->dict[str,Any]:
    models,toks,paths=load_assets(); rows=[]
    for seed in seeds:
        mapcal=pair_manifest('validation',seed,toks,'map_fit'); routesel=pair_manifest('validation',seed,toks,'route'); test=pair_manifest('test',seed,toks,'test')
        mapsets=[]; routes=[]; tfit=time.perf_counter()
        for i,p in enumerate(PAIRS):
            m=fit_maps(p,mapcal[i],models); mapsets.append(m); routes.append(choose_routes(m,routesel[i],models))
        fit_seconds=time.perf_counter()-tfit
        built=make_variants(mapsets,routes,rank,q,seed+rank*100+q); ev=evaluate(built['variants'],routes,test,mapsets,models)
        decoded={}; decode_seconds={}; roundtrip_bank_error={}
        for name,raw in built['serial'].items():
            td=time.perf_counter(); decoded_name,decoded_bank,decoded_routes=decode_payload(raw); decode_seconds[name]=time.perf_counter()-td
            if decoded_name!=name or any(not np.array_equal(a,b) for a,b in zip(routes,decoded_routes)):
                raise AssertionError(f'{name} payload metadata/route roundtrip mismatch')
            roundtrip_bank_error[name]=float(np.max(np.abs(decoded_bank-built['variants'][name])))
            if roundtrip_bank_error[name]>2e-4: raise AssertionError(f'{name} map payload roundtrip error {roundtrip_bank_error[name]}')
            decoded[name]=decoded_bank
        evrt=evaluate(decoded,routes,test,mapsets,models)
        roundtrip_nll_error={name:float(np.max(np.abs(np.asarray(ev['methods'][name]['nll'])-np.asarray(evrt['methods'][name]['nll'])))) for name in decoded}
        nroutes=nearest_routes(mapsets); nearest,_,_=pack_selected(mapsets,nroutes); env=evaluate({'nearest':nearest},nroutes,test,mapsets,models)
        artifact=Path(__file__).resolve().parents[1]/'artifacts'; artifact.mkdir(exist_ok=True)
        artifact_paths={}
        for name,raw in built['serial'].items():
            path=artifact/f'{name}_seed_{seed}.bin'; path.write_bytes(raw); artifact_paths[name]=str(path.relative_to(Path(__file__).resolve().parents[2]))
        pairrows=[]
        for pi,pair in enumerate(PAIRS):
            sl=slice(pi*12,(pi+1)*12)
            pairrows.append({'pair':list(pair),'native_nll':float(np.mean(ev['native_nll'][sl])),'native_replay_nll':float(np.mean(ev['native_replay_nll'][sl])),
                'nearest_depth_nll':float(np.mean(env['methods']['nearest']['nll'][sl])),
                'full_nll':float(np.mean(ev['methods']['full']['nll'][sl])),'pca_nll':float(np.mean(ev['methods']['pca']['nll'][sl])),'cp_nll':float(np.mean(ev['methods']['cp']['nll'][sl])),
                'route_rows':[x['row'] for x in routesel[pi]],'test_rows':[x['row'] for x in test[pi]]})
        rows.append({'seed':seed,'pair_results':pairrows,'prompt_rows':{'map_fit':[[x['row'] for x in z] for z in mapcal],'route':[[x['row'] for x in z] for z in routesel],'test':[[x['row'] for x in z] for z in test]},
            'full_mean_nll':ev['methods']['full']['mean_nll'],'pca_mean_nll':ev['methods']['pca']['mean_nll'],'cp_mean_nll':ev['methods']['cp']['mean_nll'],'native_mean_nll':ev['native_mean_nll'],
            'nearest_depth_mean_nll':env['methods']['nearest']['mean_nll'],'native_identity_max_abs_nll_diff':ev['identity_max_abs_nll_diff'],
            'payload_roundtrip':{'max_map_abs_error':roundtrip_bank_error,'max_nll_abs_error':roundtrip_nll_error,'decode_seconds':decode_seconds},
            'route_table':[[x.tolist() for x in rr] for rr in routes],'payload_bytes':{k:len(v) for k,v in built['serial'].items()},
            'payload_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in built['serial'].items()},'artifact_paths':artifact_paths,
            'compute':{'mapfit_route_seconds':fit_seconds,'cp_fit':built['cp_fit'],'application_seconds':{k:float(np.mean(v['apply_seconds'])) for k,v in ev['methods'].items()},'replay_seconds':{k:float(np.mean(v['replay_seconds'])) for k,v in ev['methods'].items()}}})
    return {'experiment_id':'MA-882','selected_pca_rank':rank,'selected_cp_rank':q,'seeds':list(seeds),'rows':rows,'models':provenance(paths),
            'runtime':{'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
