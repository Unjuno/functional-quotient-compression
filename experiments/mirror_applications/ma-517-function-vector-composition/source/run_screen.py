#!/usr/bin/env python3
"""Compose inverse relation task vectors on pinned Pythia using explicit and code products."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
D=512;RANKS=(2,4,8,12);PAIR_IDS=((0,1),(4,5),(6,7),(8,9),(12,13),(14,15))
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
COMMON_SHA='a6efcff225c4fea194ed86ce12903c2893412dffdc77674f1b3347a7df08bc88'
REVISION='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
def load_common():
 root=Path(__file__).resolve().parents[4]
 p=root/'experiments/mirror_applications/ma-516-function-vector-mirror-compression/source/run_screen.py'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=COMMON_SHA: raise ValueError('MA-516 shared extractor provenance mismatch')
 spec=importlib.util.spec_from_file_location('ma516_common',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m

def aligned_manifest(seed,common):
 manifest=[None]*len(common.TASKS)
 for a,b in PAIR_IDS:
  support_a,eval_a=common.split_task(a,seed)
  inv=dict(common.TASKS[b][1])
  support_b=[(y,inv[y]) for _,y in support_a]
  eval_b=[(y,inv[y]) for _,y in eval_a]
  assert all(inv[y]==x for x,y in support_a+eval_a)
  manifest[a]=dict(task_id=a,name=common.TASKS[a][0],support=support_a,evaluation=eval_a)
  manifest[b]=dict(task_id=b,name=common.TASKS[b][0],support=support_b,evaluation=eval_b)
 for tid,(name,_) in enumerate(common.TASKS):
  if manifest[tid] is None:
   support,queries=common.split_task(tid,seed)
   manifest[tid]=dict(task_id=tid,name=name,support=support,evaluation=queries)
 return manifest

def extract_aligned_vectors(model,tok,torch,seed,common):
 manifest=aligned_manifest(seed,common);vectors=[];token_count=0;fwd=0
 for entry in manifest:
  support=entry['support'];ds=[]
  for q in support:
   demos=[x for x in support if x[0]!=q[0]]
   full,nf=common.get_activation(model,tok,torch,common.prompt(q[0],demos))
   base,nb=common.get_activation(model,tok,torch,common.prompt(q[0]))
   token_count+=nf+nb;fwd+=2;ds.append(full-base)
  vectors.append(np.mean(ds,axis=0,dtype=np.float64).astype(np.float32))
 return np.stack(vectors),manifest,dict(forward_calls=fwd,input_tokens=token_count)

def common_payload_bytes(model_dir):
 names=('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json')
 return sum((Path(model_dir)/n).stat().st_size for n in names)

def explicit_vectors(vectors,pair):
 a,b=pair
 return dict(first=vectors[a],second=vectors[b],sum=vectors[a]+vectors[b],
             difference=vectors[a]-vectors[b],hadamard=vectors[a]*vectors[b])

def code_product(vectors,rank,common):
 mean,basis,codes,decoded=common.pca_fit(vectors,rank)
 pairs=np.asarray(PAIR_IDS,np.int16)
 out=np.stack([basis@(codes[a]*codes[b]) for a,b in PAIR_IDS]).astype(np.float32)
 return out,(mean,basis,codes)

def score_method(model,tok,torch,manifest,vectors,method_vectors):
 bypair=[];gold=[];margins=[];correct=0;total=0;tokens=0
 for pi,(a,b) in enumerate(PAIR_IDS):
  queries=manifest[a]['evaluation'];candidates=sorted({x for x,_ in queries})
  hits=0;gold_pair=[];margin_pair=[]
  for key,_ in queries:
   scores,nseq,ntok=COMMON.score_candidates(model,tok,torch,key,candidates,method_vectors[pi])
   tokens+=ntok;target=candidates.index(key);hits+=int(int(np.argmax(scores))==target)
   gold_pair.append(float(scores[target]));margin_pair.append(float(scores[target]-np.max(np.delete(scores,target))))
  acc=hits/len(queries);bypair.append(dict(first_task=a,second_task=b,accuracy=acc,mean_gold_logprob=float(np.mean(gold_pair)),mean_margin=float(np.mean(margin_pair))))
  correct+=hits;total+=len(queries);gold.extend(gold_pair);margins.extend(margin_pair)
 return dict(top1_accuracy=correct/total,mean_gold_candidate_logprob=float(np.mean(gold)),
             mean_gold_vs_best_negative_margin=float(np.mean(margins)),queries=total,
             candidate_forward_calls=total,candidate_sequences=total*8,candidate_input_tokens=tokens,by_composition=bypair)

def serialize(path,kind,vectors,rank,parts,operator_code):
 if kind=='none':return 0
 pair_ids=np.asarray(PAIR_IDS,np.int16)
 base=dict(pair_ids=pair_ids,model_sha256=np.frombuffer(MODEL_SHA.encode(),dtype='S64'),
           model_revision=np.frombuffer(REVISION.encode(),dtype='S40'),layer=np.array([4],np.int16),
           schema=np.array([517,operator_code,rank],np.int32))
 if kind==1:arrays=dict(function_vectors=vectors,**base)
 else:
  mean,basis,codes=parts
  # Same stored algebra and schema for Mirror/native product control.
  arrays=dict(shared_mean=mean,shared_basis=basis,function_codes=codes,**base)
 np.savez(path,**arrays);return Path(path).stat().st_size

def run(seed,model_dir,out):
 global COMMON
 COMMON=load_common()
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 model,tok,torch=COMMON.load_model(model_dir);base_bytes=common_payload_bytes(model_dir)
 start=time.perf_counter();vectors,manifest,support=extract_aligned_vectors(model,tok,torch,seed,COMMON);extract_s=time.perf_counter()-start
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 np.savez(out/'operand_fvs.npz',function_vectors=vectors,task_ids=np.arange(16,dtype=np.int16))
 explicit=explicit_vectors(vectors,PAIR_IDS[0]);methods=[('none',0,None),('first_fv',1,None),('second_fv',2,None),('raw_sum',3,None),('raw_difference',4,None),('raw_hadamard',5,None)]
 for r in RANKS:methods += [(f'mirror_code_product_r{r}',6,(r,'mirror')),(f'native_code_product_r{r}',7,(r,'native'))]
 rows=[]
 for label,kind,param in methods:
  compose_start=time.perf_counter()
  rank=param[0] if param else 0
  if kind==0: method_vectors=np.zeros((len(PAIR_IDS),D),np.float32);payload_kind='none';parts=None
  elif kind in (1,2,3,4,5):
   keys=('first','second','sum','difference','hadamard')
   method_vectors=np.stack([explicit_vectors(vectors,pair)[keys[kind-1]] for pair in PAIR_IDS])
   payload_kind='explicit';parts=None
  else:
   method_vectors,parts=code_product(vectors,rank,COMMON);payload_kind='code'
  compose_s=time.perf_counter()-compose_start
  op_codes={'first_fv':1,'second_fv':2,'raw_sum':3,'raw_difference':4,'raw_hadamard':5}
  operator_code=op_codes.get(label,6 if payload_kind=='code' else 0)
  payload=serialize(out/(label+'.npz'),0 if kind==0 else 1 if payload_kind=='explicit' else 2,vectors,rank,parts,operator_code)
  t=time.perf_counter();metrics=score_method(model,tok,torch,manifest,vectors,method_vectors);inference_s=time.perf_counter()-t
  if kind==0:ops=0
  elif kind in (1,2):ops=48*8*D
  elif kind in (3,4,5):ops=48*8*D+len(PAIR_IDS)*D
  else:ops=48*8*D+len(PAIR_IDS)*(rank+rank*D)
  # Audit method-specific coefficient/output norms and source vector norms.
  rows.append(dict(seed=seed,method=label,rank=rank,payload_bytes=payload,total_deployment_bytes=base_bytes+payload,
                   common_base_bytes=base_bytes,extra_compute_proxy=ops,support_examples=128,
                   support_forward_calls=support['forward_calls'],support_input_tokens=support['input_tokens'],
                   extraction_seconds=extract_s,inference_seconds=inference_s,metrics=metrics,
                   composed_vector_norms=np.linalg.norm(method_vectors,axis=1).tolist(),
                   native_alias=False))
 for r in RANKS:
  a=next(x for x in rows if x['method']==f'mirror_code_product_r{r}')
  b=next(x for x in rows if x['method']==f'native_code_product_r{r}')
  assert a['payload_bytes']==b['payload_bytes'] and a['metrics']==b['metrics'] and a['composed_vector_norms']==b['composed_vector_norms']
  b['native_alias']=True
 report=dict(experiment_id='MA-517',seed=seed,model='EleutherAI/pythia-70m-deduped',revision=REVISION,
             model_sha256=MODEL_SHA,common_base_bytes=base_bytes,extraction_seconds=extract_s,rows=rows)
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'rows':len(rows),'extraction_seconds':extract_s,
                   'summary':[dict(method=x['method'],bytes=x['payload_bytes'],accuracy=x['metrics']['top1_accuracy'],gold_lp=x['metrics']['mean_gold_candidate_logprob'],native_alias=x['native_alias']) for x in rows]},ensure_ascii=False,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
