#!/usr/bin/env python3
"""MA-547: gated residual activation edits versus sparse output-row edits."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from pinned_tasks import MODEL_SHA,CONFIG_SHA,TOKENIZER_SHA,REVISION,load_model,enc

KEYS=[f'KEY-{i:02d}' for i in range(16)]
SUPPORT_TEMPLATES=["For {key} state the assigned code:","Retrieve the code attached to {key}.","What code was assigned to {key} ?","Read the value for {key} :"]
QUERY_TEMPLATES=["Which code belongs to {key} ?","Return the stored value for {key} .","Lookup {key} . Code:","Value for {key} :"]
MARGIN_GAIN=2.0

def split_world(seed,tok):
 rng=np.random.default_rng(seed);vocab=len(tok);special=set(tok.all_special_ids)
 pool=np.asarray([i for i in range(100,vocab-100) if i not in special],dtype=np.int64)
 labels=rng.choice(pool,size=32,replace=False).tolist();pairs=list(zip(labels[:16],labels[16:]))
 # Use spaces around each key so the stored token subsequence is stable in all prompts.
 keys=list(KEYS)
 if len(set(keys))!=16:raise ValueError('key collision')
 support=[];queries=[]
 for i,key in enumerate(KEYS):
  support.extend({'key_id':i,'key':key,'text':t.format(key=key),'old':pairs[i][0],'new':pairs[i][1]} for t in SUPPORT_TEMPLATES)
  queries.extend({'key_id':i,'key':key,'text':t.format(key=key),'old':pairs[i][0],'new':pairs[i][1]} for t in QUERY_TEMPLATES)
 return support,queries,keys,pairs

def batches(tok,torch,texts):
 seq=[enc(tok,t) for t in texts];pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 maxlen=max(map(len,seq));ids=torch.full((len(seq),maxlen),pad,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,s in enumerate(seq):ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
 pos=torch.as_tensor([len(s)-1 for s in seq],dtype=torch.long)
 return ids,mask,pos,sum(map(len,seq)),seq

def route_text(text,key_strings):
 hits=[i for i,key in enumerate(key_strings) if key in text]
 if len(hits)>1:raise ValueError('ambiguous exact key route')
 return hits[0] if hits else -1

def route_batch(texts,key_strings):return np.asarray([route_text(text,key_strings) for text in texts],dtype=np.int64)

def capture(model,tok,torch,records):
 ids,mask,pos,tokens,seqs=batches(tok,torch,[x['text'] for x in records])
 with torch.no_grad():out=model(input_ids=ids,attention_mask=mask,output_hidden_states=True,use_cache=False)
 h=out.hidden_states[-1][torch.arange(len(pos)),pos].detach();logits=out.logits[torch.arange(len(pos)),pos].detach()
 return h,logits,seqs,tokens

def mirror_edit_vector(target_row,old_row,gain= MARGIN_GAIN):
 diff=target_row-old_row
 return gain*diff/(diff@diff).clamp_min(1e-12)

def row_edit_delta(mean_hidden,gain=MARGIN_GAIN):
 return gain*mean_hidden/(mean_hidden@mean_hidden).clamp_min(1e-12)

def fit_edits(model,tok,torch,support,hidden,logits,pairs):
 w=model.embed_out.weight.detach();vectors=[];row_deltas=[];biases=[];gaps=[]
 for i,(old,new) in enumerate(pairs):
  ix=[j for j,x in enumerate(support) if x['key_id']==i]
  hs=hidden[ix].mean(0);oldlog=logits[ix,old];newlog=logits[ix,new];gap=float((newlog-oldlog).mean().item());gaps.append(gap)
  diff=w[new]-w[old];vectors.append(mirror_edit_vector(w[new],w[old]))
  row_deltas.append(row_edit_delta(hs))
  biases.append(MARGIN_GAIN)
 return torch.stack(vectors),torch.stack(row_deltas),np.asarray(biases,dtype=np.float32),np.asarray(gaps,dtype=np.float64)

def kl_from_base(base,new,torch):
 p=torch.softmax(base,dim=-1);logp=torch.log_softmax(base,dim=-1);logq=torch.log_softmax(new,dim=-1)
 return (p*(logp-logq)).sum(-1)

def evaluate(model,tok,torch,queries,hidden,base_logits,seqs,key_sequences,pairs,vectors,row_deltas,biases):
 routed=route_batch([x['text'] for x in queries],key_sequences);truth=np.asarray([x['key_id'] for x in queries]);route_acc=float(np.mean(routed==truth))
 w=model.embed_out.weight.detach();n=len(queries);vocab=base_logits.shape[-1]
 base_margin=torch.stack([base_logits[j,x['new']]-base_logits[j,x['old']] for j,x in enumerate(queries)])
 # Method reports aggregated over edit interventions. Every intervention is evaluated on all held-out keys.
 methods={m:{'target_success':[],'target_margin_gain':[],'non_target_kl':[],'non_target_top1_retention':[],'non_target_margin_abs_shift':[]} for m in ('mirror_gated','roselora_global','roselora_gated','bias_gated')}
 t0=time.perf_counter()
 for edit,(old,new) in enumerate(pairs):
  targeted=np.where(routed==edit)[0];other=np.where(routed!=edit)[0]
  m=vectors[edit];dl=w@m
  # Activation view updates all vocabulary logits, but only for matching key prompts.
  mirror_logits=base_logits[targeted]+dl
  margin=mirror_logits[:,new]-mirror_logits[:,old]
  methods['mirror_gated']['target_success'].extend((margin>0).cpu().tolist());methods['mirror_gated']['target_margin_gain'].extend((margin-base_margin[targeted]).cpu().tolist())
  if len(other):
   unchanged=base_logits[other];kl=kl_from_base(unchanged,unchanged,torch);methods['mirror_gated']['non_target_kl'].extend(kl.cpu().tolist());methods['mirror_gated']['non_target_top1_retention'].extend([1.]*len(other));methods['mirror_gated']['non_target_margin_abs_shift'].extend([0.]*len(other))
  # Sparse RoseLoRA: change only the new-answer output row globally; same delta under exact gate for matched control.
  delta=(hidden@row_deltas[edit])
  row_logits=base_logits.clone();row_logits[:,new]+=delta
  margin=row_logits[targeted,new]-row_logits[targeted,old]
  methods['roselora_global']['target_success'].extend((margin>0).cpu().tolist());methods['roselora_global']['target_margin_gain'].extend((margin-base_margin[targeted]).cpu().tolist())
  if len(other):
   kl=kl_from_base(base_logits[other],row_logits[other],torch);methods['roselora_global']['non_target_kl'].extend(kl.cpu().tolist());methods['roselora_global']['non_target_top1_retention'].extend((base_logits[other].argmax(-1)==row_logits[other].argmax(-1)).float().cpu().tolist());marg=(row_logits[other,new]-row_logits[other,old])-(base_logits[other,new]-base_logits[other,old]);methods['roselora_global']['non_target_margin_abs_shift'].extend(marg.abs().cpu().tolist())
  row_gated=base_logits[targeted]+delta[targeted,None]*torch.nn.functional.one_hot(torch.full((len(targeted),),new),num_classes=vocab).to(base_logits.dtype)
  margin=row_gated[:,new]-row_gated[:,old]
  methods['roselora_gated']['target_success'].extend((margin>0).cpu().tolist());methods['roselora_gated']['target_margin_gain'].extend((margin-base_margin[targeted]).cpu().tolist())
  if len(other):methods['roselora_gated']['non_target_kl'].extend([0.]*len(other));methods['roselora_gated']['non_target_top1_retention'].extend([1.]*len(other));methods['roselora_gated']['non_target_margin_abs_shift'].extend([0.]*len(other))
  bias_logits=base_logits[targeted].clone();bias_logits[:,new]+=float(biases[edit]);margin=bias_logits[:,new]-bias_logits[:,old]
  methods['bias_gated']['target_success'].extend((margin>0).cpu().tolist());methods['bias_gated']['target_margin_gain'].extend((margin-base_margin[targeted]).cpu().tolist())
  if len(other):methods['bias_gated']['non_target_kl'].extend([0.]*len(other));methods['bias_gated']['non_target_top1_retention'].extend([1.]*len(other));methods['bias_gated']['non_target_margin_abs_shift'].extend([0.]*len(other))
 apply_s=time.perf_counter()-t0
 out={}
 for method,x in methods.items():
  out[method]={'edit_success_rate':float(np.mean(x['target_success'])),'mean_target_margin_gain':float(np.mean(x['target_margin_gain'])),'mean_non_target_kl':float(np.mean(x['non_target_kl'])),'max_non_target_kl':float(np.max(x['non_target_kl'])),'non_target_top1_retention':float(np.mean(x['non_target_top1_retention'])),'mean_abs_non_target_margin_shift':float(np.mean(x['non_target_margin_abs_shift']))}
 return out,route_acc,apply_s

def write_payload(path,arrays):np.savez(path,**arrays);return Path(path).stat().st_size

def run(seed,model_dir,outdir):
 out=Path(outdir);out.mkdir(parents=True,exist_ok=True);model,tok,torch=load_model(model_dir);torch.set_num_threads(5);torch.manual_seed(seed);np.random.seed(seed)
 support,queries,key_sequences,pairs=split_world(seed,tok)
 t0=time.perf_counter();hs,ls,ss,nt1=capture(model,tok,torch,support);hq,lq,qs,nt2=capture(model,tok,torch,queries);capture_s=time.perf_counter()-t0
 t0=time.perf_counter();vectors,row_deltas,biases,support_gaps=fit_edits(model,tok,torch,support,hs,ls,pairs);fit_s=time.perf_counter()-t0
 metrics,route_acc,apply_s=evaluate(model,tok,torch,queries,hq,lq,qs,key_sequences,pairs,vectors,row_deltas,biases)
 # Serialize trigger keys uniformly for all routed systems.
 key_bytes=[k.encode('utf-8') for k in key_sequences];maxkey=max(map(len,key_bytes));keyarr=np.zeros((len(key_bytes),maxkey),np.uint8);lengths=np.asarray([len(k) for k in key_bytes],np.int16)
 for i,k in enumerate(key_bytes):keyarr[i,:len(k)]=np.frombuffer(k,dtype=np.uint8)
 oldids=np.asarray([a for a,b in pairs],np.int32);newids=np.asarray([b for a,b in pairs],np.int32)
 meta={'key_strings_utf8':keyarr,'key_string_lengths':lengths,'edit_ids':np.arange(16,dtype=np.int16),'old_answer_ids':oldids,'new_answer_ids':newids,'model_sha256':np.frombuffer(MODEL_SHA.encode(),dtype='S64'),'revision':np.frombuffer(REVISION.encode(),dtype='S40'),'layer':np.array([-1],np.int16),'schema':np.array([547,1],np.int32),'seed':np.array([seed],np.int64)}
 bytes_={'mirror_gated':write_payload(out/'mirror_gated.npz',{'mirror_vectors':vectors.cpu().numpy().astype(np.float32),**meta}),'roselora_global':write_payload(out/'roselora_global.npz',{'row_deltas':row_deltas.cpu().numpy().astype(np.float32),'edit_ids':np.arange(16,dtype=np.int16),'new_answer_ids':newids,'model_sha256':meta['model_sha256'],'revision':meta['revision'],'schema':meta['schema'],'seed':meta['seed']}),'roselora_gated':write_payload(out/'roselora_gated.npz',{'row_deltas':row_deltas.cpu().numpy().astype(np.float32),**meta}),'bias_gated':write_payload(out/'bias_gated.npz',{'biases':biases,'key_strings_utf8':keyarr,'key_string_lengths':lengths,'edit_ids':np.arange(16,dtype=np.int16),'old_answer_ids':oldids,'new_answer_ids':newids,'model_sha256':meta['model_sha256'],'revision':meta['revision'],'schema':meta['schema'],'seed':meta['seed']}),'base':0}
 common=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'];base_bytes=sum((Path(model_dir)/f).stat().st_size for f in common)
 (out/'split_manifest.json').write_text(json.dumps({'support':support,'queries':queries,'key_sequences':key_sequences,'answer_pairs':pairs},ensure_ascii=False,indent=2)+'\n')
 report={'experiment_id':'MA-547','seed':seed,'revision':REVISION,'model_sha256':MODEL_SHA,'edits':16,'support_examples':len(support),'heldout_queries':len(queries),'support_and_query_tokens':nt1+nt2,'router_accuracy':route_acc,'support_base_margin_mean':float(support_gaps.mean()),'metrics':metrics,'serialized_bytes':bytes_,'standalone_bytes':{k:(base_bytes+v if k!='base' else base_bytes) for k,v in bytes_.items()},'common_model_bytes':base_bytes,'compute':{'capture_forward_seconds':capture_s,'analytic_edit_fit_seconds':fit_s,'all_method_edit_application_seconds':apply_s,'capture_support_queries':len(support)+len(queries),'mirror_vector_add_ops_proxy':len(queries)*512,'roselora_global_rank1_ops_proxy':len(queries)*512*16,'gated_bias_ops_proxy':len(queries)*16,'router_character_comparisons_proxy':int(sum(len(x['text']) for x in queries)*maxkey)},'gates':{'router_perfect':route_acc==1.,'mirror_edit_efficacy':min(x['edit_success_rate'] for x in (metrics['mirror_gated'],))>=.75 and metrics['mirror_gated']['mean_target_margin_gain']>=1.5,'mirror_locality':metrics['mirror_gated']['mean_non_target_kl']<=.01}}
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'router_accuracy':route_acc,'metrics':metrics,'bytes':bytes_,'gates':report['gates']},indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
