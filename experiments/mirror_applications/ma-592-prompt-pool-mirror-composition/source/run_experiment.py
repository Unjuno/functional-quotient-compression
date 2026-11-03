#!/usr/bin/env python3
"""MA-592 query-conditioned prompt-pool composition vs native L2P retrieval."""
from __future__ import annotations
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,GPTNeoXForCausalLM
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd';REV='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
TRAIN_SHA='9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f';TEST_SHA='d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0'
PLEN,HID,CTX,TGT,K,TEMP=8,512,64,64,2,0.1
METHODS=('zero_prompt','oracle_prompt','l2p_top1','l2p_top2','mirror_top2')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def get_articles(path,tok):
 out={};title=None;body=[]
 for line in Path(path).read_text(encoding='utf8').splitlines():
  if line.startswith(' = ') and line.endswith(' = ') and not line.startswith(' = ='):
   if title:
    ids=np.asarray(tok.encode(title+'\n'+'\n'.join(body),add_special_tokens=False),np.int64)
    if len(ids)>=1536:out[title]=ids
   title=line.strip(' =');body=[]
  elif title:body.append(line)
 if title:
  ids=np.asarray(tok.encode(title+'\n'+'\n'.join(body),add_special_tokens=False),np.int64)
  if len(ids)>=1536:out[title]=ids
 return out
def hidden_feature(model,ids):
 x=torch.tensor(ids,dtype=torch.long)[None]
 with torch.inference_mode():return model.gpt_neox(input_ids=x,use_cache=False,return_dict=True).last_hidden_state[0].mean(0).float().cpu().numpy()
def route_weights(query,keys):
 q=query/(np.linalg.norm(query)+1e-12);kn=keys/(np.linalg.norm(keys,axis=1,keepdims=True)+1e-12);sim=kn@q
 ix=np.argsort(-sim)[:K];z=sim[ix]/TEMP;w=np.exp(z-z.max());w=w/w.sum();return ix,w,sim
def mix_prompt(prompts,ix,w):return np.einsum('k,kph->ph',w,prompts[ix].astype(np.float32))
def nll(model,emb,prompt,tokens):
 x=torch.tensor(tokens,dtype=torch.long);pe=torch.tensor(prompt,dtype=torch.float32)[None];te=emb(x)[None];inp=torch.cat((pe,te),1)
 labels=torch.full((1,PLEN+len(tokens)),-100,dtype=torch.long);labels[0,PLEN+CTX:]=x[CTX:]
 with torch.inference_mode():o=model(inputs_embeds=inp,labels=labels,use_cache=False,return_dict=True)
 return float(o.loss)
def run(bank1,bank2,model_dir,data_dir,outdir):
 wall=time.perf_counter();torch.set_num_threads(4);data_dir=Path(data_dir);model_dir=Path(model_dir);outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
 if sha(model_dir/'model.safetensors')!=MODEL_SHA:raise ValueError('model hash mismatch')
 if sha(data_dir/'train.txt')!=TRAIN_SHA or sha(data_dir/'test.txt')!=TEST_SHA:raise ValueError('data hash mismatch')
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True);t0=time.perf_counter();model=GPTNeoXForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32).eval();load_s=time.perf_counter()-t0;emb=model.get_input_embeddings()
 docs=get_articles(data_dir/'train.txt',tok);reports=[]
 for seed,bpath in [(59101,Path(bank1)),(59102,Path(bank2))]:
  if not bpath.is_file():raise ValueError('missing MA-591 prompt artifact '+str(bpath))
  with np.load(bpath) as z:prompts=z['prompts'].astype(np.float32);titles=[str(x) for x in z['task_titles'].tolist()]
  if prompts.shape!=(8,PLEN,HID) or len(titles)!=8:raise ValueError('prompt bank shape/title mismatch')
  missing=[t for t in titles if t not in docs]
  if missing:raise ValueError('task article missing from corpus: '+repr(missing))
  t0=time.perf_counter();keys=np.stack([hidden_feature(model,docs[t][:64]) for t in titles]);key_s=time.perf_counter()-t0
  key_path=outdir/f'keys_{seed}.npz';key_bytes=np.savez(key_path,keys_fp16=keys.astype(np.float16),task_titles=np.asarray(titles),shape=np.asarray([8,HID],np.int32),schema=np.asarray([592,1],np.int32));key_bytes=key_path.stat().st_size
  # Keep one inference copy of the prompt bank and charge its actual serialized artifact bytes.
  pool_path=outdir/f'prompt_pool_{seed}.npz';np.savez(pool_path,prompts_fp16=prompts.astype(np.float16),task_titles=np.asarray(titles),shape=np.asarray([8,PLEN,HID],np.int32),schema=np.asarray([592,2],np.int32));pool_bytes=pool_path.stat().st_size
  nlls={m:[] for m in METHODS};route_hits=0;top2_hits=0;route_s=0.;mix_s=0.;inference={m:0. for m in METHODS};rows=[]
  for ti,title in enumerate(titles):
   ids=docs[title]
   for start in (1024,1152,1280,1408):
    seq=ids[start-CTX:start+TGT];t0=time.perf_counter();q=hidden_feature(model,seq[:CTX]);ix,w,sim=route_weights(q,keys);route_s+=time.perf_counter()-t0
    route_hits+=int(ix[0]==ti);top2_hits+=int(ti in ix)
    t0=time.perf_counter();composed=mix_prompt(prompts,ix,w);mix_s+=time.perf_counter()-t0
    prompt_set={'zero_prompt':np.zeros((PLEN,HID),np.float32),'oracle_prompt':prompts[ti],'l2p_top1':prompts[ix[0]],'l2p_top2':composed,'mirror_top2':composed}
    for method,prompt in prompt_set.items():
     t0=time.perf_counter();loss=nll(model,emb,prompt,seq);inference[method]+=time.perf_counter()-t0;nlls[method].append(loss)
    rows.append({'task':title,'top1':int(ix[0]),'top2':[int(x) for x in ix],'weights':[float(x) for x in w],'correct_task':ti,'similarities':[float(sim[x]) for x in ix]})
  vals={m:float(np.mean(v)) for m,v in nlls.items()};code_bytes_per_query=2+2*K # two uint8 pool indices plus two fp16 weights
  metrics={'bank_seed':seed,'task_titles':titles,'nll':vals,'nll_delta_vs_oracle':{m:vals[m]-vals['oracle_prompt'] for m in vals},'top1_route_accuracy':route_hits/len(rows),'top2_task_containment':top2_hits/len(rows),'prompt_pool_bytes':pool_bytes,'key_bank_bytes':key_bytes,'total_persistent_payload_bytes':pool_bytes+key_bytes,'transient_coordinate_bytes_per_query':code_bytes_per_query,'total_query_coordinate_bytes':code_bytes_per_query*len(rows),'total_payload_including_dev_query_codes':pool_bytes+key_bytes+code_bytes_per_query*len(rows),'query_count':len(rows),'routing_seconds_total':route_s,'mix_seconds_total':mix_s,'key_extraction_seconds':key_s,'inference_seconds':inference,'virtual_prompt_tokens':PLEN,'source_prompt_bank_sha256':sha(bpath),'routes':rows}
  out=(outdir/f'metrics_{seed}.json');out.write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n');reports.append(metrics)
 report={'experiment_id':'MA-592','model_revision':REV,'model_sha256':MODEL_SHA,'dataset_sha256':{'train.txt':TRAIN_SHA,'test.txt':TEST_SHA},'model_load_seconds':load_s,'banks':reports,'total_wall_seconds':time.perf_counter()-wall,'note':'Mirror top2 and native L2P top2 implement the same softmax-weighted prompt sum; exact output equality is expected and audited.'}
 (outdir/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank-59101',type=Path,required=True);p.add_argument('--bank-59102',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.bank_59101,a.bank_59102,a.model_dir,a.data_dir,a.out)
if __name__=='__main__':main()
