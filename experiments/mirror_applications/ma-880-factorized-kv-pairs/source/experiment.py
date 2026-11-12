"""CPU cross-model KV translator family screen for MA-876."""
from __future__ import annotations
import os
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import json,struct,time,resource,hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig

DEV_CAL=24
MODELS=('openai-community/gpt2','EleutherAI/pythia-70m-deduped','EleutherAI/pythia-160m-deduped','HuggingFaceTB/SmolLM2-135M')
TRAIN_PAIRS=((MODELS[0],MODELS[1]),(MODELS[1],MODELS[2]),(MODELS[2],MODELS[3]),(MODELS[3],MODELS[0]))
DEV_PAIRS=((MODELS[1],MODELS[0]),(MODELS[3],MODELS[2]))
FRESH_PAIRS=((MODELS[2],MODELS[1]),(MODELS[0],MODELS[3]))
PAIRS=TRAIN_PAIRS+DEV_PAIRS+FRESH_PAIRS
RANKS=(32,); FACTOR_RANKS=(1,2,4,8); D=64; ALPHA=.01; MIN_TOKENS=16; MAX_TOKENS=128
FRESH_SEEDS=(983,997,1009); DEV_SEED=829
HEADER=struct.Struct('<5sBBB')
ROLE_LAYER_BINS=16; ROLE_HEAD_BINS=8; ROLE_COUNT=ROLE_LAYER_BINS*ROLE_HEAD_BINS*2

def model_root()->Path: return Path.home()/'.cache/huggingface/hub'
def snapshot(model_id:str)->Path:
    p=model_root()/('models--'+model_id.replace('/','--'))/'snapshots'
    if not p.exists(): raise FileNotFoundError(f'cached local model unavailable: {model_id}')
    return sorted(p.iterdir())[0]
def load_assets()->tuple[dict[str,Any],dict[str,Any],dict[str,Path]]:
    torch.set_num_threads(min(4,os.cpu_count() or 1)); models={}; toks={}; paths={}
    for name in MODELS:
        p=snapshot(name); paths[name]=p
        toks[name]=AutoTokenizer.from_pretrained(p,local_files_only=True,use_fast=True)
        models[name]=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,attn_implementation='eager').eval()
    return models,toks,paths

def corpus(split:str): return load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split=split,download_mode='reuse_dataset_if_exists')
def eligible_prompts(split:str,pair:tuple[str,str],toks:dict[str,Any],minimum:int)->list[dict[str,Any]]:
    ds=corpus(split); st,tt=toks[pair[0]],toks[pair[1]]; out=[]; seen=set()
    for i,row in enumerate(ds):
        text=row['text'].strip()
        if len(text)<32 or text in seen: continue
        a=st(text,add_special_tokens=False,return_offsets_mapping=True); b=tt(text,add_special_tokens=False,return_offsets_mapping=True)
        if a['offset_mapping']!=b['offset_mapping']: continue
        n=len(a['input_ids'])
        if MIN_TOKENS<=n<=MAX_TOKENS:
            seen.add(text); out.append({'row':i,'text':text,'src_ids':a['input_ids'],'tgt_ids':b['input_ids'],'tokens':n})
            if len(out)>=minimum: break
    return out

def select_prompts(split:str,pair:tuple[str,str],toks:dict[str,Any],seed:int,count:int,role:str)->list[dict[str,Any]]:
    need=(DEV_CAL+16 if split=='train' else len(FRESH_SEEDS)*count)
    arr=eligible_prompts(split,pair,toks,need)
    pair_tag=sum((i+1)*ord(c) for i,c in enumerate(pair[0]+'>'+pair[1]))
    if split=='train':
        perm=np.random.default_rng(seed+pair_tag%10000).permutation(len(arr))
        start=0 if role=='cal' else DEV_CAL
        chosen=perm[start:start+count]
    else:
        # One fixed permutation, then disjoint seed-indexed chunks across fresh worlds.
        pair_tag=sum((i+1)*ord(c) for i,c in enumerate(pair[0]+'>'+pair[1]))
        perm=np.random.default_rng(51_236+pair_tag+(0 if role=='cal' else 7_919)).permutation(len(arr))
        group=FRESH_SEEDS.index(seed)
        chosen=perm[group*count:(group+1)*count]
    if len(chosen)<count: raise RuntimeError(f'need {count} aligned {split} prompts for {pair}; found {len(chosen)}')
    return [arr[int(i)] for i in chosen]

def cache_arrays(model:Any,ids:list[int])->tuple[list[tuple[np.ndarray,np.ndarray]],Any,float]:
    x=torch.tensor(ids,dtype=torch.long).unsqueeze(0)
    t0=time.perf_counter()
    with torch.inference_mode(): out=model(input_ids=x,use_cache=True)
    elapsed=time.perf_counter()-t0; past=out.past_key_values
    if hasattr(past,'to_legacy_cache'): past=past.to_legacy_cache()
    arr=[]
    for k,v in past:
        arr.append((k[0].detach().cpu().numpy().astype(np.float64),v[0].detach().cpu().numpy().astype(np.float64)))
    return arr,out,elapsed

def map_refs(src_config:Any,tgt_config:Any)->list[tuple[int,int,int,int,int]]:
    slayers=src_config.num_hidden_layers; tlayers=tgt_config.num_hidden_layers
    sheads=getattr(src_config,'num_key_value_heads',None) or src_config.num_attention_heads
    theads=getattr(tgt_config,'num_key_value_heads',None) or tgt_config.num_attention_heads
    dsrc=src_config.hidden_size//src_config.num_attention_heads; dtgt=tgt_config.hidden_size//tgt_config.num_attention_heads
    if dsrc!=D or dtgt!=D: raise ValueError(f'head dims must be {D}, got {dsrc}->{dtgt}')
    refs=[]
    for tl in range(tlayers):
        sl=round(tl*(slayers-1)/max(1,tlayers-1))
        for th in range(theads):
            sh=round(th*(sheads-1)/max(1,theads-1))
            for kv in (0,1): refs.append((tl,th,kv,sl,sh))
    return refs

def fit_pair_maps(pair:tuple[str,str],prompts:list[dict[str,Any]],models:dict[str,Any],toks:dict[str,Any])->dict[str,Any]:
    src,tgt=pair; refs=map_refs(models[src].config,models[tgt].config); stats=[]; count=0
    for _ in refs: stats.append([np.zeros((D+1,D+1),np.float64),np.zeros((D+1,D),np.float64)])
    wall0=time.perf_counter(); src_time=tgt_time=0.
    for p in prompts:
        sc,_,ts=cache_arrays(models[src],p['src_ids']); tc,_,tt=cache_arrays(models[tgt],p['tgt_ids']); src_time+=ts; tgt_time+=tt
        n=min(len(sc[0][0][0]),len(tc[0][0][0])); count+=n
        for j,(tl,th,kv,sl,sh) in enumerate(refs):
            x=sc[sl][kv][sh,:n,:]; y=tc[tl][kv][th,:n,:]
            xa=np.concatenate([x,np.ones((n,1),np.float64)],axis=1)
            stats[j][0]+=xa.T@xa; stats[j][1]+=xa.T@y
    maps=[]; eye=np.eye(D+1); eye[-1,-1]=0.
    for sxx,sxy in stats: maps.append(np.linalg.solve(sxx+ALPHA*eye,sxy))
    return {'pair':pair,'refs':refs,'maps':np.stack(maps),'samples':count,'calibration_seconds':time.perf_counter()-wall0,'source_forward_seconds':src_time,'target_forward_seconds':tgt_time}

def fit_basis(mapsets:list[dict[str,Any]])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    vals=np.concatenate([x['maps'].reshape(len(x['maps']),-1) for x in mapsets],axis=0)
    mean=vals.mean(axis=0); centered=vals-mean
    _,_,vt=np.linalg.svd(centered,full_matrices=False)
    return mean,vt,vals

def compress_maps(maps:np.ndarray,mean:np.ndarray,basis:np.ndarray,rank:int)->tuple[np.ndarray,np.ndarray]:
    flat=maps.reshape(len(maps),-1); coeff=(flat-mean)@basis[:rank].T; rec=mean+coeff@basis[:rank]
    return rec.reshape(maps.shape),coeff

def apply_maps(src_cache:list[tuple[np.ndarray,np.ndarray]],refs:list[tuple[int,int,int,int,int]],maps:np.ndarray,tgt_config:Any)->list[tuple[torch.Tensor,torch.Tensor]]:
    hkv=getattr(tgt_config,'num_key_value_heads',None) or tgt_config.num_attention_heads; layers=tgt_config.num_hidden_layers
    length=src_cache[0][0].shape[1]; outk=np.zeros((layers,hkv,length,D),np.float32); outv=np.zeros_like(outk)
    for i,(tl,th,kv,sl,sh) in enumerate(refs):
        x=src_cache[sl][kv][sh]
        xa=np.concatenate([x,np.ones((length,1),np.float64)],axis=1)
        y=(xa@maps[i]).astype(np.float32)
        (outk if kv==0 else outv)[tl,th]=y
    return [(torch.from_numpy(outk[l:l+1]),torch.from_numpy(outv[l:l+1])) for l in range(layers)]

def target_nll(model:Any,logits:torch.Tensor,label:int)->float:
    return float(F.cross_entropy(logits.reshape(1,-1),torch.tensor([label],dtype=torch.long)))

def transfer_nll(model:Any,past:list[tuple[torch.Tensor,torch.Tensor]],query:int,label:int)->tuple[float,float]:
    ids=torch.tensor([[query]],dtype=torch.long); t0=time.perf_counter()
    with torch.inference_mode(): out=model(input_ids=ids,past_key_values=tuple(past),use_cache=True)
    return target_nll(model,out.logits[0,-1],label),time.perf_counter()-t0

def eval_heldout(pair:tuple[str,str],prompts:list[dict[str,Any]],models:dict[str,Any],basis_mean:np.ndarray,basis:np.ndarray,rank:int,independent_maps:np.ndarray,refs:list[tuple[int,int,int,int,int]])->dict[str,Any]:
    src,tgt=pair; model_s,model_t=models[src],models[tgt]
    low_maps,coeff=compress_maps(independent_maps,basis_mean,basis,rank)
    native=[]; full_nll=[]; low_nll=[]; native_time=[]; src_time=[]; full_apply=[]; low_apply=[]; suffix_full=[]; suffix_low=[]; rel_full=[]; rel_low=[]; prompt_ids=[]
    for p in prompts:
        sc,_,st=cache_arrays(model_s,p['src_ids']); tc,to,tt=cache_arrays(model_t,p['tgt_ids']); src_time.append(st); native_time.append(tt)
        logits=to.logits[0,-2]; label=p['tgt_ids'][-1]; native.append(target_nll(model_t,logits,label))
        # Relative heldout cache error, using all positions in the aligned prompt.
        full_start=time.perf_counter(); full_past=apply_maps(sc,refs,independent_maps,model_t.config); full_apply.append(time.perf_counter()-full_start)
        low_start=time.perf_counter(); low_past=apply_maps(sc,refs,low_maps,model_t.config); low_apply.append(time.perf_counter()-low_start)
        def relerr(predpast:list[tuple[torch.Tensor,torch.Tensor]])->float:
            num=den=0.
            for l,(pk,pv) in enumerate(predpast):
                tk,tv=tc[l]
                for x,y in ((pk[0].numpy(),tk),(pv[0].numpy(),tv)):
                    num+=float(np.sum((x-y)**2)); den+=float(np.sum(y**2))
            return num/max(den,1e-30)
        rel_full.append(relerr(full_past)); rel_low.append(relerr(low_past))
        # Prefix excludes query and label; query token is target tokenizer's penultimate ID.
        q=p['tgt_ids'][-2]; label=p['tgt_ids'][-1]; prefix_len=len(p['tgt_ids'])-2
        def trim(past): return [(k[:,:,:prefix_len,:].contiguous(),v[:,:,:prefix_len,:].contiguous()) for k,v in past]
        x,y=transfer_nll(model_t,trim(full_past),q,label); full_nll.append(x); suffix_full.append(y)
        x,y=transfer_nll(model_t,trim(low_past),q,label); low_nll.append(x); suffix_low.append(y)
        prompt_ids.append(p['row'])
    return {'native_nll':native,'full_map_nll':full_nll,'compressed_nll':low_nll,'full_map_nll_delta':float(np.mean(full_nll)-np.mean(native)),'compressed_nll_delta':float(np.mean(low_nll)-np.mean(native)),'relative_cache_mse_full':float(np.mean(rel_full)),'relative_cache_mse_compressed':float(np.mean(rel_low)),'source_prefill_seconds':src_time,'target_native_prefill_seconds':native_time,'full_translation_seconds':full_apply,'compressed_translation_seconds':low_apply,'target_suffix_seconds_full':suffix_full,'target_suffix_seconds_compressed':suffix_low,'prompt_rows':prompt_ids,'map_codes':coeff.tolist()}

def serialize_family_payloads(mapsets:list[dict[str,Any]],mean:np.ndarray,basis:np.ndarray,rank:int)->tuple[bytes,bytes]:
    names=[(a+'>'+b).encode() for a,b in (x['pair'] for x in mapsets)]
    metas=b'\0'.join(names)+b'\0'; all_refs=[r for x in mapsets for r in x['refs']]
    meta=bytes(v for r in all_refs for v in r)
    full=bytearray(HEADER.pack(b'MA876',0,rank,len(mapsets))); full.extend(metas)
    shared=bytearray(HEADER.pack(b'MA876',1,rank,len(mapsets))); shared.extend(metas)
    full_maps=np.concatenate([x['maps'] for x in mapsets],axis=0)
    full.extend(np.asarray(full_maps,dtype='<f4').tobytes()); shared.extend(np.asarray(mean,dtype='<f4').tobytes()); shared.extend(np.asarray(basis[:rank],dtype='<f4').tobytes())
    for x in mapsets:
        _,codes=compress_maps(x['maps'],mean,basis,rank); shared.extend(np.asarray(codes,dtype='<f4').tobytes())
    full.extend(meta); shared.extend(meta)
    return bytes(full),bytes(shared)

def serialize_family(mapsets:list[dict[str,Any]],mean:np.ndarray,basis:np.ndarray,rank:int)->tuple[int,int]:
    full,shared=serialize_family_payloads(mapsets,mean,basis,rank)
    return len(full),len(shared)

def file_provenance(paths:dict[str,Path])->dict[str,Any]:
    out={}
    for name,p in paths.items():
        weight=p/'model.safetensors'; token_files=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']
        out[name]={'snapshot':p.name,'weights_bytes':weight.stat().st_size if weight.exists() else 0,'tokenizer_config_bytes':sum((p/x).stat().st_size for x in token_files if (p/x).exists())}
    return out

def run_world(seed:int,cal_split:str,test_split:str,ncal:int,ntest:int,ranks:tuple[int,...],models:dict[str,Any],toks:dict[str,Any])->dict[str,Any]:
    all_maps=[]; prompt_manifests={}; t0=time.perf_counter()
    for pi,pair in enumerate(PAIRS):
        cal=select_prompts(cal_split,pair,toks,seed,ncal,'cal'); prompt_manifests[f'{pair[0]}>{pair[1]}_cal']=[p['row'] for p in cal]
        all_maps.append(fit_pair_maps(pair,cal,models,toks))
    cal_wall=time.perf_counter()-t0
    bt=time.perf_counter(); mean,basis,_=fit_basis(all_maps[:2]); basis_seconds=time.perf_counter()-bt; held=all_maps[2]
    test=select_prompts(test_split,HELDOUT,toks,seed,ntest,'test'); prompt_manifests[f'{HELDOUT[0]}>{HELDOUT[1]}_test']=[p['row'] for p in test]
    rank_eval={}
    for rank in ranks:
        rank_eval[str(rank)]=eval_heldout(HELDOUT,test,models,mean,basis,rank,held['maps'],held['refs'])
    full_bytes,shared_bytes=serialize_family(all_maps,mean,basis,max(ranks))
    return {'seed':seed,'calibration_seconds':cal_wall,'basis_fit_seconds':basis_seconds,'map_fit_seconds':[x['calibration_seconds'] for x in all_maps],'basis_train_maps':sum(len(x['maps']) for x in all_maps[:2]),'heldout_maps':len(held['maps']),'rank_results':rank_eval,'family_full_bytes':full_bytes,'family_shared_bytes_by_rank':{str(r):serialize_family(all_maps,mean,basis,r)[1] for r in ranks},'heldout_full_map_bytes':len(held['maps'])*(D+1)*D*4+len(held['refs'])*5,'heldout_shared_marginal_bytes':{str(r):len(mean.astype('<f4').tobytes())+len(basis[:r].astype('<f4').tobytes())+len(held['maps'])*r*4+len(held['refs'])*5 for r in ranks},'prompt_rows':prompt_manifests,'mean_tokens_calibration':[x['samples']/ncal for x in all_maps]}

def run_dev()->dict[str,Any]:
    models,toks,paths=load_assets(); row=run_world(DEV_SEED,'train','train',24,16,RANKS,models,toks)
    return {'experiment_id':'MA-876','split':'wikitext train','ranks':list(RANKS),'row':row,'models':file_provenance(paths),'runtime':{'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}

def run_fresh(seeds:tuple[int,...],rank:int)->dict[str,Any]:
    models,toks,paths=load_assets(); rows=[run_world(s,'validation','test',12,16,(rank,),models,toks) for s in seeds]
    return {'experiment_id':'MA-876','seeds':list(seeds),'rank':rank,'rows':rows,'models':file_provenance(paths),'runtime':{'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}

def write_json(path:str|Path,data:dict[str,Any])->None: Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


# MA-880: canonical map PCA plus a CP source-model × target-model × role factor.
def role_index(ref:tuple[int,int,int,int,int],target_config:Any)->int:
    tl,th,kv,_,_=ref
    nlayer=target_config.num_hidden_layers
    nheads=getattr(target_config,'num_key_value_heads',None) or target_config.num_attention_heads
    li=int(round(tl*(ROLE_LAYER_BINS-1)/max(1,nlayer-1)))
    hi=int(round(th*(ROLE_HEAD_BINS-1)/max(1,nheads-1)))
    return (li*ROLE_HEAD_BINS+hi)*2+kv

def factor_observations(mapsets:list[dict[str,Any]],mean:np.ndarray,basis:np.ndarray,rank:int,models:dict[str,Any]):
    src_ids=[]; tgt_ids=[]; roles=[]; codes=[]; pair_tags=[]
    id_of={name:i for i,name in enumerate(MODELS)}
    for item in mapsets:
        src,tgt=item['pair']; flat=item['maps'].reshape(len(item['maps']),-1)
        code=(flat-mean)@basis[:rank].T
        for j,ref in enumerate(item['refs']):
            src_ids.append(id_of[src]); tgt_ids.append(id_of[tgt])
            roles.append(role_index(ref,models[tgt].config)); codes.append(code[j]); pair_tags.append(f'{src}>{tgt}')
    return np.asarray(src_ids),np.asarray(tgt_ids),np.asarray(roles),np.asarray(codes,dtype=np.float32),pair_tags

def fit_pair_factor(mapsets:list[dict[str,Any]],mean:np.ndarray,basis:np.ndarray,rank:int,q:int,steps:int,
                    models:dict[str,Any],seed:int)->dict[str,Any]:
    si,ti,ri,y,pair_tags=factor_observations(mapsets,mean,basis,rank,models)
    torch.manual_seed(seed); dtype=torch.float32
    idx_s=torch.as_tensor(si); idx_t=torch.as_tensor(ti); idx_r=torch.as_tensor(ri); target=torch.as_tensor(y)
    u=torch.nn.Parameter(torch.randn((len(MODELS),q),dtype=dtype)*.1)
    v=torch.nn.Parameter(torch.randn((len(MODELS),q),dtype=dtype)*.1)
    role=torch.nn.Parameter(torch.randn((ROLE_COUNT,q,rank),dtype=dtype)*.1)
    bias=torch.nn.Parameter(torch.zeros(rank,dtype=dtype))
    opt=torch.optim.Adam([u,v,role,bias],lr=.01)
    start=time.perf_counter(); losses=[]
    for _ in range(steps):
        pred=bias+(u[idx_s]*v[idx_t]).unsqueeze(-1).mul(role[idx_r]).sum(dim=1)
        loss=torch.mean((pred-target)**2)+1e-5*(u.square().mean()+v.square().mean()+role.square().mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    return {'u':u.detach().numpy(),'v':v.detach().numpy(),'role':role.detach().numpy(),'bias':bias.detach().numpy(),
            'q':q,'rank':rank,'steps':steps,'fit_seconds':time.perf_counter()-start,
            'train_code_mse':float(np.mean((bias.detach().numpy()+(u.detach().numpy()[si]*v.detach().numpy()[ti])[:,:,None].__mul__(role.detach().numpy()[ri]).sum(axis=1)-y)**2)),
            'last_loss':losses[-1],'train_pairs':sorted(set(pair_tags))}

def predict_pair_maps(pair:tuple[str,str],factor:dict[str,Any],mean:np.ndarray,basis:np.ndarray,
                      refs:list[tuple[int,int,int,int,int]],models:dict[str,Any])->np.ndarray:
    si=MODELS.index(pair[0]); ti=MODELS.index(pair[1]);
    bins=np.asarray([role_index(ref,models[pair[1]].config) for ref in refs],dtype=np.int64)
    codes=factor['bias'][None,:]+((factor['u'][si]*factor['v'][ti])[None,:,None]*factor['role'][bins]).sum(axis=1)
    return (mean[None,:]+codes@basis[:factor['rank']]).reshape(len(refs),D+1,D).astype(np.float64)

def evaluate_maps(pair:tuple[str,str],prompts:list[dict[str,Any]],models:dict[str,Any],
                  full_maps:np.ndarray,factor_maps:np.ndarray,pca_maps:np.ndarray,
                  refs:list[tuple[int,int,int,int,int]])->dict[str,Any]:
    src,tgt=pair; ms,mt=models[src],models[tgt]
    metrics={name:{'nll':[],'rel_cache_mse':[],'apply_seconds':[],'suffix_seconds':[]} for name in ('full','factor','pca')}
    native=[]; native_prefill=[]; source_prefill=[]; rows=[]; tokens=0
    for p in prompts:
        sc,_,st=cache_arrays(ms,p['src_ids']); tc,to,tt=cache_arrays(mt,p['tgt_ids'])
        source_prefill.append(st); native_prefill.append(tt); tokens+=p['tokens']; rows.append(p['row'])
        native.append(target_nll(mt,to.logits[0,-2],p['tgt_ids'][-1]))
        plen=len(p['tgt_ids'])-2; query=p['tgt_ids'][-2]; label=p['tgt_ids'][-1]
        for name,maps in (('full',full_maps),('factor',factor_maps),('pca',pca_maps)):
            t0=time.perf_counter(); past=apply_maps(sc,refs,maps,mt.config); metrics[name]['apply_seconds'].append(time.perf_counter()-t0)
            pk=[(k[:,:,:plen,:].contiguous(),v[:,:,:plen,:].contiguous()) for k,v in past]
            nll_value,suffix=transfer_nll(mt,pk,query,label); metrics[name]['nll'].append(nll_value); metrics[name]['suffix_seconds'].append(suffix)
            num=den=0.0
            for layer,(k,v) in enumerate(past):
                tk,tv=tc[layer]; tk=tk[:,:plen,:]; tv=tv[:,:plen,:]
                num+=float(np.sum((k[0].numpy()-tk)**2)+np.sum((v[0].numpy()-tv)**2))
                den+=float(np.sum(tk**2)+np.sum(tv**2))
            metrics[name]['rel_cache_mse'].append(num/max(den,1e-30))
    for name in metrics:
        metrics[name]['mean_nll']=float(np.mean(metrics[name]['nll']))
        metrics[name]['delta_native']=metrics[name]['mean_nll']-float(np.mean(native))
        metrics[name]['mean_rel_cache_mse']=float(np.mean(metrics[name]['rel_cache_mse']))
    return {'pair':list(pair),'native_nll':native,'native_mean_nll':float(np.mean(native)),
            'native_prefill_seconds':native_prefill,'source_prefill_seconds':source_prefill,
            'metrics':metrics,'prompt_rows':rows,'test_tokens':tokens}

def serialize_factor_payload(factor:dict[str,Any],mean:np.ndarray,basis:np.ndarray,
                             outputs:list[dict[str,Any]])->bytes:
    names=b'\0'.join(n.encode() for n in MODELS)+b'\0'
    pair_meta=b'\0'.join((x['pair'][0]+'>'+x['pair'][1]).encode() for x in outputs)+b'\0'
    refs=bytes(v for item in outputs for ref in item['refs'] for v in ref)
    payload=bytearray(struct.pack('<5sBBBB',b'MA880',1,factor['rank'],factor['q'],len(outputs)))
    payload.extend(names); payload.extend(pair_meta); payload.extend(refs)
    for arr in (mean,basis[:factor['rank']],factor['bias'],factor['u'],factor['v'],factor['role']):
        payload.extend(np.asarray(arr,dtype='<f4').tobytes())
    return bytes(payload)

def factor_payload_bytes(factor:dict[str,Any],mean:np.ndarray,basis:np.ndarray,
                         outputs:list[dict[str,Any]])->int:
    return len(serialize_factor_payload(factor,mean,basis,outputs))

def full_payload_bytes(outputs:list[dict[str,Any]])->int:
    names=b'\0'.join((x['pair'][0]+'>'+x['pair'][1]).encode() for x in outputs)+b'\0'
    refs=bytes(v for item in outputs for ref in item['refs'] for v in ref)
    maps=np.concatenate([x['maps'] for x in outputs],axis=0)
    payload=bytearray(struct.pack('<5sBBBB',b'MA880',0,0,0,len(outputs))); payload.extend(names)
    payload.extend(np.asarray(maps,dtype='<f4').tobytes()); payload.extend(refs)
    return len(payload)

def run_ma880_dev()->dict[str,Any]:
    models,toks,paths=load_assets(); start=time.perf_counter()
    train=[]
    for pair in TRAIN_PAIRS:
        prompts=select_prompts('train',pair,toks,DEV_SEED,24,'cal'); train.append(fit_pair_maps(pair,prompts,models,toks))
    mean,basis,_=fit_basis(train); candidates={}
    for q in FACTOR_RANKS:
        factor=fit_pair_factor(train,mean,basis,32,q,600,models,DEV_SEED+q)
        pair_scores=[]
        for pair in DEV_PAIRS:
            cal=select_prompts('train',pair,toks,DEV_SEED,24,'cal')
            full=fit_pair_maps(pair,cal,models,toks)
            test=select_prompts('train',pair,toks,DEV_SEED,16,'test')
            pm=predict_pair_maps(pair,factor,mean,basis,full['refs'],models)
            pca,_=compress_maps(full['maps'],mean,basis,32)
            ev=evaluate_maps(pair,test,models,full['maps'],pm,pca,full['refs'])
            pair_scores.append({'pair':list(pair),'factor_mean_nll':ev['metrics']['factor']['mean_nll'],
                                'native_mean_nll':ev['native_mean_nll'],'full_mean_nll':ev['metrics']['full']['mean_nll'],
                                'pca_mean_nll':ev['metrics']['pca']['mean_nll']})
        candidates[str(q)]={'factor':factor,'dev_pairs':pair_scores,
                             'mean_factor_nll':float(np.mean([x['factor_mean_nll'] for x in pair_scores]))}
    chosen=min(FACTOR_RANKS,key=lambda q:(candidates[str(q)]['mean_factor_nll'],q))
    return {'experiment_id':'MA-880','dev_seed':DEV_SEED,'selected_q':chosen,'selected_steps':600,
            'selection_rule':'lowest mean next-token NLL over two whole ordered dev pairs; tie smaller CP rank',
            'train_pairs':[list(x) for x in TRAIN_PAIRS],'dev_pairs':[list(x) for x in DEV_PAIRS],
            'factor_rank_candidates':list(FACTOR_RANKS),'candidate_results':{q:{'q':int(q),'fit_seconds':v['factor']['fit_seconds'],
             'train_code_mse':v['factor']['train_code_mse'],'mean_factor_nll':v['mean_factor_nll'],'dev_pairs':v['dev_pairs']}
             for q,v in candidates.items()},'models':file_provenance(paths),
            'runtime':{'cuda_available':torch.cuda.is_available(),'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                       'total_seconds':time.perf_counter()-start}}

def run_ma880_fresh(seeds:tuple[int,...],q:int,steps:int=600)->dict[str,Any]:
    models,toks,paths=load_assets(); rows=[]
    for seed in seeds:
        train=[]; train_prompt_rows={}
        for pair in TRAIN_PAIRS:
            prompts=select_prompts('validation',pair,toks,seed,12,'cal')
            train_prompt_rows[f'{pair[0]}>{pair[1]}']=[p['row'] for p in prompts]
            train.append(fit_pair_maps(pair,prompts,models,toks))
        mean,basis,_=fit_basis(train); factor=fit_pair_factor(train,mean,basis,32,q,steps,models,seed+7717)
        outputs=[]; evals=[]; manifests={'train_calibration':train_prompt_rows,'heldout_calibration':{},'heldout_test':{}}
        for pair in FRESH_PAIRS:
            cal=select_prompts('validation',pair,toks,seed,12,'cal')
            manifests['heldout_calibration'][f'{pair[0]}>{pair[1]}']=[p['row'] for p in cal]
            full=fit_pair_maps(pair,cal,models,toks); outputs.append(full)
            test=select_prompts('test',pair,toks,seed,12,'test')
            manifests['heldout_test'][f'{pair[0]}>{pair[1]}']=[p['row'] for p in test]
            fmaps=predict_pair_maps(pair,factor,mean,basis,full['refs'],models)
            pca,_=compress_maps(full['maps'],mean,basis,32)
            evals.append(evaluate_maps(pair,test,models,full['maps'],fmaps,pca,full['refs']))
        factor_payload=serialize_factor_payload(factor,mean,basis,outputs)
        full_bytes=full_payload_bytes(outputs)
        artifact_dir=Path(__file__).resolve().parents[1]/'artifacts'; artifact_dir.mkdir(exist_ok=True)
        payload_path=artifact_dir/f'factor_payload_seed_{seed}.bin'; payload_path.write_bytes(factor_payload)
        rows.append({'seed':seed,'pair_results':evals,'prompt_rows':manifests,'factor':{k:v for k,v in factor.items() if k not in ('u','v','role','bias')},
                     'training_map_fit_seconds':[x['calibration_seconds'] for x in train],
                     'training_map_samples':[x['samples'] for x in train],
                     'factor_payload_bytes':len(factor_payload),'independent_full_payload_bytes':full_bytes,
                     'factor_bytes_saved_fraction':1-len(factor_payload)/full_bytes,
                     'factor_payload_sha256':hashlib.sha256(factor_payload).hexdigest(),
                     'factor_payload_path':str(payload_path.relative_to(Path(__file__).resolve().parents[2]))})
    return {'experiment_id':'MA-880','seeds':list(seeds),'q':q,'steps':steps,'rows':rows,
            'models':file_provenance(paths),'runtime':{'cuda_available':torch.cuda.is_available(),
            'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}}
