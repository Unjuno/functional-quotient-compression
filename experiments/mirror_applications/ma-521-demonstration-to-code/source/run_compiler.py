#!/usr/bin/env python3
"""Frozen MA-521 demo-to-code compiler screen."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HARNESS=Path(__file__).with_name('shared_fv_harness.py')
spec=importlib.util.spec_from_file_location('ma516_shared_fv_harness',HARNESS)
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
RANK=3;FIT=np.arange(12);HELD=np.arange(12,16);MODEL_HASH=h.MODEL_SHA;REVISION=h.REVISION

def demo_features(model,tok,torch,manifest):
    emb=model.get_input_embeddings();features=[];token_counts=[];flat=[];offsets=[0]
    with torch.no_grad():
        for task in manifest:
            ids_all=[]
            for x,y in task['support']:
                ids_all.extend(h.enc(tok,f'Input: {x}\nOutput: {y}'))
            if not ids_all:raise ValueError('empty support encoding')
            ids=torch.tensor(ids_all,dtype=torch.long)
            v=emb(ids).mean(dim=0).detach().cpu().numpy().astype(np.float32)
            v=v/(np.linalg.norm(v)+1e-12)
            features.append(v);token_counts.append(len(ids_all));flat.extend(ids_all);offsets.append(len(flat))
    return np.stack(features),np.asarray(token_counts,dtype=np.int64),np.asarray(flat,dtype=np.int32),np.asarray(offsets,dtype=np.int64)

def fit_compiler(features,targets,seed,nonlinear,updates=2000):
    import torch
    torch.manual_seed(seed)
    gen=torch.Generator(device='cpu');gen.manual_seed(seed)
    x=torch.as_tensor(features[FIT],dtype=torch.float32);y_np=targets[FIT].astype(np.float32)
    mean=y_np.mean(axis=0,dtype=np.float64).astype(np.float32);y=torch.as_tensor(y_np-mean)
    A=torch.nn.Parameter(torch.randn((512,RANK),generator=gen)/np.sqrt(512.0))
    ab=torch.nn.Parameter(torch.zeros(RANK))
    B=torch.nn.Parameter(torch.randn((512,RANK),generator=gen)/np.sqrt(512.0))
    params=[A,ab,B];opt=torch.optim.Adam(params,lr=.01,weight_decay=0.)
    t=time.perf_counter()
    for _ in range(updates):
        opt.zero_grad(set_to_none=True)
        code=x@A+ab
        if nonlinear:code=torch.tanh(code)
        pred=code@B.T
        loss=((pred-y)**2).mean();loss.backward();opt.step()
    fit_s=time.perf_counter()-t
    a=A.detach().cpu().numpy().astype(np.float32);bias=ab.detach().cpu().numpy().astype(np.float32);basis=B.detach().cpu().numpy().astype(np.float32)
    codes=encode_features(features,a,bias,nonlinear)
    decoded=mean+codes@basis.T
    return a,bias,basis,mean,codes,decoded.astype(np.float32),fit_s

def encode_features(features,A,bias,nonlinear):
    z=features@A+bias
    return np.tanh(z).astype(np.float32) if nonlinear else z.astype(np.float32)

def evaluate_direct_icl(model,tok,torch,manifest):
    total=correct=0;gold=[];margins=[];calls=seqs=tokens=0;per_task=[]
    for tid in HELD:
        task=manifest[int(tid)];demos=task['support'];queries=task['evaluation'];cands=sorted({y for _,y in queries})
        tc=0;tg=[];tm=[]
        for x,y in queries:
            scores,nc,nt=score_with_demos(model,tok,torch,x,cands,demos,None)
            gi=cands.index(y);tc+=int(int(np.argmax(scores))==gi);tg.append(float(scores[gi]));tm.append(float(scores[gi]-np.max(np.delete(scores,gi))))
            calls+=1;seqs+=len(cands);tokens+=nt
        per_task.append({'task_id':int(tid),'accuracy':tc/len(queries),'mean_gold_logprob':float(np.mean(tg)),'mean_gold_margin':float(np.mean(tm))})
        correct+=tc;total+=len(queries);gold.extend(tg);margins.extend(tm)
    return {'heldout_accuracy':correct/total,'mean_gold_candidate_logprob':float(np.mean(gold)),
      'mean_gold_vs_best_negative_margin':float(np.mean(margins)),'queries':total,'candidate_forward_calls':calls,
      'candidate_sequences':seqs,'candidate_input_tokens':tokens,'by_task':per_task}

def score_with_demos(model,tok,torch,query,candidates,demos,vec):
    prefix=h.prompt(query,demos);pids=h.enc(tok,prefix);seqs=[];targets=[]
    for c in candidates:
        full=h.enc(tok,prefix+' '+c)
        if full[:len(pids)]!=pids:raise ValueError('candidate tokenization changed the frozen prompt prefix')
        seqs.append(full);targets.append(full[len(pids):])
    maxlen=max(map(len,seqs));pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids=torch.full((len(seqs),maxlen),pad,dtype=torch.long);mask=torch.zeros_like(ids)
    for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
    logits=h.hook_forward(model,torch,ids,mask,len(pids)-1,vec);lp=torch.log_softmax(logits,dim=-1);scores=[]
    for i,ans in enumerate(targets):
        sc=0.
        for j,tokid in enumerate(ans):sc+=float(lp[i,len(pids)+j-1,tokid].item())
        scores.append(sc)
    return np.asarray(scores,dtype=np.float64),len(seqs),sum(map(len,seqs))

def eval_vector(model,tok,torch,vectors,manifest):return h.evaluate(model,tok,torch,vectors,vectors,manifest)

def eval_compiled(model,tok,torch,decoded,manifest):return h.evaluate(model,tok,torch,decoded,decoded,manifest)

def payload(path,method,vectors,features,parts,token_ids,offsets):
    common=dict(task_ids=np.arange(16,dtype=np.int16),layer=np.array([h.LAYER_OUT],np.int16),
      model_revision=np.frombuffer(REVISION.encode(),dtype='S40'),model_sha256=np.frombuffer(MODEL_HASH.encode(),dtype='S64'),
      schema=np.array([521,RANK],np.int32))
    if method=='none':return 0
    if method=='direct_icl':arrays=dict(demo_token_ids=token_ids,demo_offsets=offsets,**common)
    elif method=='explicit_fv':arrays=dict(function_vectors=vectors.astype(np.float32),**common)
    else:
        A,ab,B,mean,codes=parts
        arrays=dict(encoder_weight=A,encoder_bias=ab,decoder_basis=B,decoder_mean=mean,function_codes=codes,
          **common)
    np.savez(path,**arrays);size=Path(path).stat().st_size
    with zipfile.ZipFile(path) as z:
        if any(i.compress_type!=zipfile.ZIP_STORED for i in z.infolist()):raise AssertionError('NPZ must be uncompressed')
    return size

def vector_audit(ref,dec):
    x=ref[HELD].astype(np.float64);y=dec[HELD].astype(np.float64);e=x-y
    c=np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)+1e-12)
    return {'heldout_relative_rmse':float(np.linalg.norm(e)/np.linalg.norm(x)),
      'heldout_cosine_mean':float(np.mean(c)),'heldout_cosine_min':float(np.min(c))}

def run(seed,model_dir,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);model,tok,torch=h.load_model(model_dir)
    t=time.perf_counter();vectors,manifest,support=h.extract_task_vectors(model,tok,torch,seed);extract_s=time.perf_counter()-t
    feature_t=time.perf_counter();features,feat_tokens,token_ids,offsets=demo_features(model,tok,torch,manifest);feature_s=time.perf_counter()-feature_t
    np.savez(out/'teacher_vectors.npz',task_vectors=vectors,task_ids=np.arange(16,dtype=np.int16))
    np.savez(out/'demo_features.npz',features=features,feature_token_counts=feat_tokens)
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    basefiles=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    basebytes=sum((Path(model_dir)/n).stat().st_size for n in basefiles)
    compile_results={}
    for name,nonlin in [('native_linear_compiler',False),('mirror_tanh_compiler',True)]:
        A,ab,B,mean,codes,decoded,fit_s=fit_compiler(features,vectors,seed,nonlin)
        compile_results[name]=(decoded,(A,ab,B,mean,codes),fit_s,codes)
    rows=[]
    methods=['none','direct_icl','explicit_fv','native_linear_compiler','mirror_tanh_compiler']
    for name in methods:
        if name=='none':decoded=np.zeros_like(vectors);parts=None;updates=0;fit_s=method_feature_s=0.
        elif name=='direct_icl':decoded=np.zeros_like(vectors);parts=None;updates=0;fit_s=0.;method_feature_s=0.
        elif name=='explicit_fv':decoded=vectors;parts=None;updates=0;fit_s=method_feature_s=0.
        else:decoded,parts,fit_s,_=compile_results[name];updates=2000;method_feature_s=feature_s
        pbytes=payload(out/(name+'.npz'),name,vectors,features,parts,token_ids,offsets)
        if name=='direct_icl':t=time.perf_counter();metrics=evaluate_direct_icl(model,tok,torch,manifest);infer_s=time.perf_counter()-t
        else:t=time.perf_counter();metrics=eval_compiled(model,tok,torch,decoded,manifest);infer_s=time.perf_counter()-t
        va=vector_audit(vectors,decoded) if name in ('native_linear_compiler','mirror_tanh_compiler') else {}
        saved_codes=compile_results[name][3] if name in compile_results else np.zeros((16,RANK),np.float32)
        train_proxy=2000*12*512*RANK*2 if updates else 0
        compile_proxy=int(len(token_ids)*512 + 16*512*RANK*2) if name in ('native_linear_compiler','mirror_tanh_compiler') else 0
        query_proxy=(512*metrics['candidate_sequences'] if name not in ('none','direct_icl') else 0)
        total_s=extract_s+method_feature_s+fit_s+infer_s
        row={'seed':seed,'method':name,'rank':RANK if 'compiler' in name else 0,'payload_bytes':pbytes,
         'common_base_bytes':basebytes,'total_deployment_bytes':basebytes+pbytes,'support_examples':128,
         'support_forward_calls':support['forward_calls'],'support_input_tokens':support['input_tokens'],
         'demonstration_feature_tokens':int(np.sum(feat_tokens)),'candidate_input_tokens':metrics['candidate_input_tokens'],
         'optimizer_updates':updates,'extraction_seconds':extract_s,'feature_seconds':method_feature_s,'fit_seconds':fit_s,
         'inference_seconds':infer_s,'wall_time_seconds':total_s,'active_compute_proxy':train_proxy+compile_proxy+query_proxy,
         'metrics':metrics,'vector_metrics':va,'query_context_tokens_per_query':metrics['candidate_input_tokens']/max(metrics['candidate_sequences'],1)}
        rows.append(row)
    report={'experiment_id':'MA-521','seed':seed,'model':'EleutherAI/pythia-70m-deduped','revision':REVISION,'model_sha256':MODEL_HASH,
      'dimension':512,'common_base_bytes':basebytes,'fit_task_ids':FIT.tolist(),'heldout_task_ids':HELD.tolist(),
      'extraction_seconds':extract_s,'feature_seconds':feature_s,'rows':rows}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'rows':[{'method':r['method'],'bytes':r['payload_bytes'],'accuracy':r['metrics']['heldout_accuracy'],
      'gold_logprob':r['metrics']['mean_gold_candidate_logprob'],'tokens':r['metrics']['candidate_input_tokens'],**r['vector_metrics']} for r in rows]},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
