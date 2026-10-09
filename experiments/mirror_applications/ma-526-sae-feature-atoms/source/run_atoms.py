#!/usr/bin/env python3
"""Run the frozen MA-526 SAE shared-feature-pool screen."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];HARNESS=Path(__file__).with_name('shared_fv_harness.py')
spec=importlib.util.spec_from_file_location('ma516_shared_fv_harness',HARNESS);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
RANK=8;POOL_SIZE=64;SAE_SHA='85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4';SAE_REV='36c2027509efad69836aa4231999c9b717848ecd';FIT=np.arange(12);HELD=np.arange(12,16)
class TiedSAE(torch.nn.Module):
    """Safe module shell for the released checkpoint's tied-SAE parameters."""
    def __init__(self):super().__init__()

def load_sae(path,torch):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SAE_SHA:raise ValueError('pinned SAE hash mismatch')
    torch.serialization.add_safe_globals([TiedSAE])
    sae=torch.load(path,map_location='cpu',weights_only=True)
    W=sae.encoder.detach().cpu().numpy().astype(np.float32)
    bias=sae.encoder_bias.detach().cpu().numpy().astype(np.float32)
    if W.shape!=(2048,512) or bias.shape!=(2048,):raise ValueError('unexpected tied-SAE shape')
    return W,bias

def sae_encode(vectors,W,bias):return np.maximum(vectors@W.T+bias,0).astype(np.float32)

def select_pool(vectors,W,bias,fit_ids=FIT):
    z=sae_encode(vectors[fit_ids],W,bias)
    score=z.sum(axis=0)
    return np.argsort(-score,kind='stable')[:POOL_SIZE].astype(np.int16),score

def matching_pursuit(vectors,atoms,k=RANK):
    norms=np.linalg.norm(atoms.astype(np.float64),axis=1)
    if np.any(norms<=1e-12):raise ValueError('zero SAE atom')
    dn=(atoms/norms[:,None]).astype(np.float32)
    idxs=np.zeros((len(vectors),k),dtype=np.int16);vals=np.zeros((len(vectors),k),dtype=np.float32);decoded=np.zeros_like(vectors,dtype=np.float32)
    for i,v in enumerate(vectors.astype(np.float32)):
        residual=v.copy();used=np.zeros(len(atoms),dtype=bool)
        for j in range(k):
            corr=dn@residual;corr[used]=0
            ix=int(np.argmax(np.abs(corr)))
            if abs(float(corr[ix]))<1e-12:break
            coef=float(corr[ix]/norms[ix]);idxs[i,j]=ix;vals[i,j]=coef;used[ix]=True
            atom=atoms[ix];decoded[i]+=coef*atom;residual-=coef*atom
    return idxs,vals,decoded

def native_topk(vectors,W,bias,k=RANK):
    z=sae_encode(vectors,W,bias);ids=np.argpartition(-z,kth=k-1,axis=1)[:,:k]
    vals=np.take_along_axis(z,ids,axis=1);decoded=np.zeros_like(vectors,dtype=np.float32)
    for i in range(len(vectors)):decoded[i]=vals[i]@W[ids[i]]
    return ids.astype(np.int16),vals.astype(np.float32),decoded

def save_payload(path,method,metadata,parts=None):
    if method=='none':return 0
    if method=='explicit_fv':arrays=dict(function_vectors=parts['vectors'],**metadata)
    elif method=='native_sae_top8':arrays=dict(feature_ids=parts['ids'],coefficients=parts['values'],**metadata)
    elif method=='global_omp8':arrays=dict(feature_ids=parts['ids'],coefficients=parts['values'],**metadata)
    else:arrays=dict(pool_feature_ids=parts['pool'].astype(np.int16),code_indices=parts['ids'].astype(np.uint8),coefficients=parts['values'],**metadata)
    np.savez(path,**arrays);n=Path(path).stat().st_size
    with zipfile.ZipFile(path) as z:
        if any(i.compress_type!=zipfile.ZIP_STORED for i in z.infolist()):raise AssertionError('payload must be uncompressed NPZ')
    return n

def vector_metrics(ref,dec):
    x=ref[HELD].astype(np.float64);y=dec[HELD].astype(np.float64);e=x-y;c=np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)+1e-12)
    return {'heldout_relative_rmse':float(np.linalg.norm(e)/np.linalg.norm(x)),'heldout_cosine_mean':float(np.mean(c)),'heldout_cosine_min':float(np.min(c))}

def run(seed,model_dir,sae_path,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);model,tok,torch=h.load_model(model_dir)
    t=time.perf_counter();vectors,manifest,support=h.extract_task_vectors(model,tok,torch,seed);extract_s=time.perf_counter()-t
    W,bias=load_sae(sae_path,torch);coding_start=time.perf_counter()
    pool,pool_scores=select_pool(vectors,W,bias)
    pidx,pvals,pdec=matching_pursuit(vectors,W[pool],RANK)
    # Convert shared-pool-local indices back only at decode time; serialization keeps uint8 local IDs.
    tidx,tvals,tdec=native_topk(vectors,W,bias,RANK)
    gidx,gvals,gdec=matching_pursuit(vectors,W,RANK)
    coding_s=time.perf_counter()-coding_start
    basefiles=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    model_bytes=sum((Path(model_dir)/n).stat().st_size for n in basefiles);sae_bytes=Path(sae_path).stat().st_size
    metadata=dict(task_ids=np.arange(16,dtype=np.int16),model_revision=np.frombuffer(h.REVISION.encode(),dtype='S40'),
      model_sha256=np.frombuffer(h.MODEL_SHA.encode(),dtype='S64'),sae_revision=np.frombuffer(SAE_REV.encode(),dtype='S40'),
      sae_sha256=np.frombuffer(SAE_SHA.encode(),dtype='S64'),layer=np.array([h.HOOK_LAYER],np.int16),schema=np.array([526,2048,RANK,POOL_SIZE],np.int32))
    explicit_payload_bytes=save_payload(out/'explicit_fv.npz','explicit_fv',metadata,{'vectors':vectors.astype(np.float32)})
    methods=[('none',np.zeros_like(vectors),None),('explicit_fv',vectors,{'vectors':vectors.astype(np.float32)}),
      ('native_sae_top8',tdec,{'ids':tidx,'values':tvals}),('global_omp8',gdec,{'ids':gidx,'values':gvals}),
      ('mirror_pool64_omp8',pdec,{'pool':pool,'ids':pidx,'values':pvals})]
    rows=[]
    for name,decoded,parts in methods:
        pbytes=save_payload(out/(name+'.npz'),name,metadata,parts)
        start=time.perf_counter();metrics=h.evaluate(model,tok,torch,vectors,decoded,manifest);infer_s=time.perf_counter()-start
        vm=vector_metrics(vectors,decoded) if name not in ('none','explicit_fv') else {}
        uses_sae=name not in ('none','explicit_fv');common=model_bytes+(sae_bytes if uses_sae else 0)
        seqs=metrics['candidate_sequences'];infer_ops=512*seqs if name not in ('none','explicit_fv') else 0
        if name=='mirror_pool64_omp8':coding_ops=12*512*2048+16*RANK*512*POOL_SIZE
        elif name=='global_omp8':coding_ops=16*RANK*512*2048
        elif name=='native_sae_top8':coding_ops=16*512*2048
        else:coding_ops=0
        standalone=model_bytes+(sae_bytes if uses_sae else 0)+pbytes
        row={'seed':seed,'method':name,'payload_bytes':pbytes,'model_bytes':model_bytes,'sae_bytes':sae_bytes if uses_sae else 0,
          'common_base_bytes':common,'total_deployment_bytes':standalone,'incremental_over_preloaded_sae_bytes':pbytes,
          'explicit_fv_total_bytes':model_bytes+explicit_payload_bytes,
          'support_examples':128,'support_forward_calls':support['forward_calls'],'support_input_tokens':support['input_tokens'],
          'candidate_input_tokens':metrics['candidate_input_tokens'],'optimizer_updates':0,'extraction_seconds':extract_s,
          'coding_seconds':coding_s if name not in ('none','explicit_fv') else 0.,'inference_seconds':infer_s,
          'wall_time_seconds':extract_s+(coding_s if name not in ('none','explicit_fv') else 0.)+infer_s,
          'active_compute_proxy':int(coding_ops+infer_ops),'metrics':metrics,'vector_metrics':vm,
          'nonzero_coefficients_per_task':int(np.count_nonzero(parts['values'])/len(vectors)) if name in ('native_sae_top8','global_omp8','mirror_pool64_omp8') else 0}
        rows.append(row)
    np.savez(out/'sae_dictionary_audit.npz',pool_ids=pool,pool_scores=pool_scores)
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report={'experiment_id':'MA-526','seed':seed,'model':'EleutherAI/pythia-70m-deduped','revision':h.REVISION,
      'model_sha256':h.MODEL_SHA,'sae_repo':'Elriggs/pythia-70M-deduped-sae','sae_revision':SAE_REV,'sae_sha256':SAE_SHA,
      'sae_bytes':sae_bytes,'sae_shape':[2048,512],'model_bytes':model_bytes,'pool_feature_ids':pool.tolist(),
      'pool_selected_from_task_ids':FIT.tolist(),'heldout_task_ids':HELD.tolist(),'pool_size':POOL_SIZE,'sparse_budget':RANK,
      'pool_selection_top_aggregate_encoder_activation':True,'extraction_seconds':extract_s,'coding_seconds':coding_s,'rows':rows}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'pool_ids':pool.tolist(),'rows':[{'method':r['method'],'payload_bytes':r['payload_bytes'],'total_bytes':r['total_deployment_bytes'],'accuracy':r['metrics']['heldout_accuracy'],'gold_logprob':r['metrics']['mean_gold_candidate_logprob'],**r['vector_metrics']} for r in rows]},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--sae',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.sae,a.out)
if __name__=='__main__':main()
