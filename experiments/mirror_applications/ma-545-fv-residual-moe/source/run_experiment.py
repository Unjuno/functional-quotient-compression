#!/usr/bin/env python3
"""MA-545: few-shot-context routed function-vector experts."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from task_data_and_pinning import TASKS, MODEL_SHA, CONFIG_SHA, TOKENIZER_SHA, REVISION, split_task, load_model, enc, prompt

LAYER_OUT=4
HOOK_LAYER=3
D_MODEL=512
ROUTER_UPDATES=1000

def demo_set(support, query):
    return [(x,y) for x,y in support[:4] if x != query]

def prompt_with_query(query, demos=()):
    lines=[f'Input: {x}\nOutput: {y}' for x,y in demos]
    lines.append(f'Input: {query}\nOutput:')
    return '\n'.join(lines)

def get_hidden(model,tok,torch,text):
    ids=enc(tok,text); x=torch.tensor([ids],dtype=torch.long)
    with torch.no_grad(): out=model(input_ids=x,output_hidden_states=True,use_cache=False)
    return out.hidden_states[LAYER_OUT][0,-1].detach().cpu().numpy().astype(np.float32),len(ids)

def extract(model,tok,torch,seed):
    vectors=[]; manifest=[]; router_x=[]; route_y=[]; tokens=0; calls=0
    for tid,(name,pairs) in enumerate(TASKS):
        support,queries=split_task(tid,seed)
        diffs=[]
        for qx,qy in support:
            demos=[p for p in support if p[0] != qx]
            full,nf=get_hidden(model,tok,torch,prompt_with_query(qx,demos))
            base,nb=get_hidden(model,tok,torch,prompt_with_query(qx))
            diffs.append(full-base); tokens+=nf+nb; calls+=2
            # Router sees the frozen four-shot query context. Remove query if duplicated.
            feat,n=get_hidden(model,tok,torch,prompt_with_query(qx,demo_set(support,qx)))
            router_x.append(feat); route_y.append(tid); tokens+=n; calls+=1
        vectors.append(np.mean(diffs,axis=0,dtype=np.float64).astype(np.float32))
        manifest.append({'task_id':tid,'name':name,'support':support,'queries':queries})
    return np.stack(vectors),np.stack(router_x),np.asarray(route_y,dtype=np.int64),manifest,{'forward_calls':calls,'input_tokens':tokens}

def fit_router(x,y,torch):
    mean=x.mean(0);std=x.std(0);std[std<1e-6]=1.
    xn=torch.tensor((x-mean)/std,dtype=torch.float32)
    yt=torch.tensor(y,dtype=torch.long)
    layer=torch.nn.Linear(D_MODEL,len(TASKS));opt=torch.optim.Adam(layer.parameters(),lr=.01,weight_decay=0.)
    for _ in range(ROUTER_UPDATES):
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(layer(xn),yt);loss.backward();opt.step()
    w=layer.weight.detach().cpu().numpy().astype(np.float32);b=layer.bias.detach().cpu().numpy().astype(np.float32)
    return {'mean':mean.astype(np.float32),'std':std.astype(np.float32),'weight':w,'bias':b}

def route(router,x):
    z=(x-router['mean'])/router['std']
    logits=z@router['weight'].T+router['bias']
    return np.argmax(logits,axis=1),logits

def capture_mlp_input(model,torch,input_ids):
    layer=model.gpt_neox.layers[HOOK_LAYER]; box={}
    def pre(module,args): box['x']=args[0].detach()[0,-1].cpu().numpy().astype(np.float32)
    h=layer.mlp.dense_h_to_4h.register_forward_pre_hook(pre)
    try:
        with torch.no_grad(): model(input_ids=input_ids,use_cache=False)
    finally: h.remove()
    return box['x']

def fit_rank1_controls(model,tok,torch,vectors,manifest):
    As=[];Bs=[];counts=[];ridge=1e-3
    for tid,task in enumerate(manifest):
        xs=[]
        for qx,qy in task['support']:
            demos=demo_set(task['support'],qx);ids=torch.tensor([enc(tok,prompt_with_query(qx,demos))],dtype=torch.long)
            xs.append(capture_mlp_input(model,torch,ids))
        x=np.stack(xs).astype(np.float64)
        target=np.broadcast_to(vectors[tid].astype(np.float64),(len(x),D_MODEL))
        # Ridge minimum-norm map; retain only its best rank-one SVD component.
        w=target.T@x@np.linalg.inv(x.T@x+ridge*np.eye(D_MODEL))
        u,s,vt=np.linalg.svd(w,full_matrices=False)
        B=(u[:,0]*np.sqrt(max(s[0],0))).astype(np.float32)
        A=(vt[0,:]*np.sqrt(max(s[0],0))).astype(np.float32)
        As.append(A);Bs.append(B);counts.append(len(xs))
    return np.stack(As),np.stack(Bs),{'support_examples':sum(counts),'ridge':ridge}

def hooked_logits(model,torch,ids,mask,pos,vector=None,lora=None):
    layer=model.gpt_neox.layers[HOOK_LAYER];handle=None
    if vector is not None:
        v=torch.as_tensor(vector,dtype=torch.float32)
        def add(module,args,out):
            h=out[0].clone();h[:,pos,:]+=v;return (h,)+tuple(out[1:])
        handle=layer.register_forward_hook(add)
    elif lora is not None:
        a,b=lora; at=torch.as_tensor(a,dtype=torch.float32);bt=torch.as_tensor(b,dtype=torch.float32)
        def add_lora(module,args,out):
            x=args[0][:,pos,:]
            delta=(x@at).unsqueeze(-1)*bt
            h=out.clone();h[:,pos,:]+=delta.squeeze(-1)
            return h
        # Hook the layer MLP output itself; its input is the pre-MLP residual tensor.
        handle=layer.mlp.register_forward_hook(add_lora)
    try:
        with torch.no_grad(): return model(input_ids=ids,attention_mask=mask,use_cache=False).logits
    finally:
        if handle is not None: handle.remove()

def score(model,tok,torch,query,candidates,demos,vector=None,lora=None):
    prefix=prompt_with_query(query,demos);pids=enc(tok,prefix);seqs=[];targets=[]
    for c in candidates:
        full=enc(tok,prefix+' '+c)
        if full[:len(pids)]!=pids: raise ValueError('candidate tokenization changed prefix')
        seqs.append(full);targets.append(full[len(pids):])
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    maxlen=max(map(len,seqs));ids=torch.full((len(seqs),maxlen),pad,dtype=torch.long);mask=torch.zeros_like(ids)
    for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
    logits=hooked_logits(model,torch,ids,mask,len(pids)-1,vector,lora);lp=torch.log_softmax(logits,dim=-1);scores=[]
    for i,ans in enumerate(targets):
        val=0.
        for j,t in enumerate(ans):val+=float(lp[i,len(pids)+j-1,t].item())
        scores.append(val)
    return np.asarray(scores,dtype=np.float64),len(seqs),sum(map(len,seqs))

def evaluate(model,tok,torch,vectors,router,router_x,router_y,manifest,As,Bs,seed):
    pred,_=route(router,router_x); route_acc=float(np.mean(pred==router_y))
    methods=['icl','shared_mean','oracle_fv','routed_fv','routed_rank1']
    stats={m:{'correct':0,'queries':0,'gold':[],'margin':[],'sequences':0,'tokens':0} for m in methods}
    t0=time.perf_counter()
    for tid,task in enumerate(manifest):
        candidates=sorted({y for _,y in task['queries']})
        for qi,(x,gold) in enumerate(task['queries']):
            demos=demo_set(task['support'],x)
            # Router query feature is generated from the same held-out prompt and never uses its answer.
            rfeat,_=get_hidden(model,tok,torch,prompt_with_query(x,demos));routed,_=route(router,rfeat[None,:]);rid=int(routed[0])
            for m in methods:
                vec=None;lr=None
                if m=='shared_mean':vec=vectors.mean(0)
                elif m=='oracle_fv':vec=vectors[tid]
                elif m=='routed_fv':vec=vectors[rid]
                elif m=='routed_rank1':lr=(As[rid],Bs[rid])
                sc,n,nt=score(model,tok,torch,x,candidates,demos,vec,lr)
                gi=candidates.index(gold);st=stats[m];st['correct']+=int(np.argmax(sc)==gi);st['queries']+=1;st['gold'].append(float(sc[gi]));st['margin'].append(float(sc[gi]-np.max(np.delete(sc,gi))));st['sequences']+=n;st['tokens']+=nt
    elapsed=time.perf_counter()-t0
    out={}
    for m,s in stats.items():out[m]={'accuracy':s['correct']/s['queries'],'mean_gold_logprob':float(np.mean(s['gold'])),'mean_gold_margin':float(np.mean(s['margin'])),'queries':s['queries'],'candidate_sequences':s['sequences'],'candidate_tokens':s['tokens']}
    out['route_accuracy_support']=route_acc
    out['heldout_route_accuracy']=None
    out['evaluation_seconds']=elapsed
    return out

def npz_bytes(path,arrays):
    np.savez(path,**arrays);return Path(path).stat().st_size

def run(seed,model_dir,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    model,tok,torch=load_model(model_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)
    start=time.perf_counter();fvs,rx,ry,manifest,excompute=extract(model,tok,torch,seed);extract_s=time.perf_counter()-start
    router=fit_router(rx,ry,torch);As,Bs,wmeta=fit_rank1_controls(model,tok,torch,fvs,manifest)
    metrics=evaluate(model,tok,torch,fvs,router,rx,ry,manifest,As,Bs,seed)
    # Route held-out query prompts for accuracy (kept distinct from support train metric).
    rpred=[];rytrue=[]
    for tid,task in enumerate(manifest):
        for x,y in task['queries']:
            feat,_=get_hidden(model,tok,torch,prompt_with_query(x,demo_set(task['support'],x)))
            pred,_=route(router,feat[None,:]);rpred.append(int(pred[0]));rytrue.append(tid)
    metrics['heldout_route_accuracy']=float(np.mean(np.asarray(rpred)==np.asarray(rytrue)))
    common=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    base_bytes=sum((Path(model_dir)/f).stat().st_size for f in common)
    manifest_bytes=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    metadata={'task_ids':np.arange(len(TASKS),dtype=np.int16),'layer':np.array([HOOK_LAYER],dtype=np.int16),'revision':np.frombuffer(REVISION.encode(),dtype='S40'),'model_sha256':np.frombuffer(MODEL_SHA.encode(),dtype='S64'),'schema':np.array([545,1],dtype=np.int32),'support_split_manifest_utf8':np.frombuffer(manifest_bytes,dtype=np.uint8)}
    payloads={}
    payloads['oracle_fv']=npz_bytes(outdir/'oracle_fv.npz',{'function_vectors':fvs,**metadata})
    payloads['routed_fv']=npz_bytes(outdir/'routed_fv.npz',{'router_mean':router['mean'],'router_std':router['std'],'router_weight':router['weight'],'router_bias':router['bias'],'function_vectors':fvs,**metadata})
    payloads['rank1']=npz_bytes(outdir/'rank1.npz',{'router_mean':router['mean'],'router_std':router['std'],'router_weight':router['weight'],'router_bias':router['bias'],'lora_A':As,'lora_B':Bs,**metadata})
    payloads['shared_mean']=npz_bytes(outdir/'shared_mean.npz',{'shared_vector':fvs.mean(0),**metadata})
    payloads['icl']=0
    (outdir/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    np.savez(outdir/'extract.npz',function_vectors=fvs,router_x=rx,router_y=ry,lora_A=As,lora_B=Bs)
    report={'experiment_id':'MA-545','seed':seed,'revision':REVISION,'model_sha256':MODEL_SHA,'metrics':metrics,'payload_bytes':payloads,'standalone_bytes':{k:(base_bytes+v if k!='icl' else base_bytes) for k,v in payloads.items()},'common_model_bytes':base_bytes,'compute':{'extraction_seconds':extract_s,'evaluation_seconds':metrics['evaluation_seconds'],'support_examples':len(TASKS)*8,'query_examples':len(TASKS)*8,'extract_forward_calls':excompute['forward_calls'],'extract_input_tokens':excompute['input_tokens'],'router_updates':ROUTER_UPDATES,'router_fit_mac_proxy':ROUTER_UPDATES*len(rx)*D_MODEL*len(TASKS),'candidate_sequences':metrics['routed_fv']['candidate_sequences'],'candidate_tokens':metrics['routed_fv']['candidate_tokens'],'FV_add_dims':metrics['routed_fv']['queries']*D_MODEL,'rank1_active_rank_ops':metrics['routed_rank1']['queries']*2*D_MODEL},'weight_control_fit':wmeta}
    (outdir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'route':metrics['heldout_route_accuracy'],'accuracy':{k:v['accuracy'] for k,v in metrics.items() if isinstance(v,dict) and 'accuracy' in v},'bytes':payloads},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
