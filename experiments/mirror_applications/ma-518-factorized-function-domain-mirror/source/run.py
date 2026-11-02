import csv, hashlib, io, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; PAY=ART/'payloads'
MODEL='openai-community/gpt2'; REV='607a30d783dfa663caf39e06633721c8d4cfcd7e'
DEV=[51800,51801]; FRESH=[51810,51811,51812]; SEEDS=[0,1,2]
PROCS=['prepend','append','reverse','rotate']; DOMAINS={'fruit':['apple','banana','cherry','lemon'],'shape':['circle','square','triangle','diamond'],'color':['red','blue','green','yellow'],'vehicle':['car','truck','train','plane']}
PAIR_ALL=[(p,d) for p in range(4) for d in range(4)]
# Hold out diagonal pairs; train on the complementary 12 pairs.
TRAIN=[(p,d) for p,d in PAIR_ALL if p!=d]; HELD=[(i,i) for i in range(4)]
RANK=4; LAYER=6; torch.set_num_threads(4)

def load():
 tok=AutoTokenizer.from_pretrained(MODEL,revision=REV);tok.padding_side='left';tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,torch_dtype=torch.float32);model.eval();return model,tok

def token_sanity(tok):
 vals=[]
 for words in DOMAINS.values(): vals += words
 vals += ['The','is','operation','fruit','shape','color','vehicle','result','input','output']
 return {w:tok(' '+w,add_special_tokens=False)['input_ids'] for w in vals}

def demos(p,d):
 words=DOMAINS[d]; proc=PROCS[p]
 # Each procedure is a deterministic one-token operation over a four-item ordered domain.
 lines=[]
 for j,w in enumerate(words[:3]):
  if proc=='prepend': y=words[0]
  elif proc=='append': y=words[-1]
  elif proc=='reverse': y=words[3-j]
  else: y=words[(j+1)%4]
  lines.append(f'For {d}, {proc} {w} gives {y}.\n')
 return ''.join(lines)

def query(p,d):
 return f'For {d}, {PROCS[p]} {DOMAINS[d][3]} gives'

def item_vec(model,tok,p,d):
 pre='Task mapping:\n'
 prompts=[pre+'Apply:',pre+demos(p,d)+query(p,d)]
 b=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False)
 with torch.no_grad():o=model(**b,output_hidden_states=True,return_dict=True)
 return (o.hidden_states[LAYER+1][1,-1]-o.hidden_states[LAYER+1][0,-1]).detach()

def logits(model,tok,prompts,vecs=None):
 b=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False); h=None
 if vecs is not None:
  def hook(_m,_i,out):
   x=out[0] if isinstance(out,tuple) else out;x=x.clone();x[:,-1,:]+=vecs.to(x.device);return (x,)+out[1:] if isinstance(out,tuple) else x
  h=model.transformer.h[LAYER].register_forward_hook(hook)
 try:
  with torch.no_grad():return model(**b,return_dict=True).logits[:,-1,:]
 finally:
  if h is not None:h.remove()

def payload(obj):
 b=io.BytesIO();torch.save(obj,b);return b.getvalue()

def dev():
 model,tok=load(); sanity=token_sanity(tok)
 multi={k:v for k,v in sanity.items() if len(v)!=1}
 assert not multi, f'not single-token: {multi}'
 assert len(TRAIN)==12 and len(HELD)==4 and not(set(TRAIN)&set(HELD))
 assert all((p,d) not in TRAIN for p,d in HELD)
 # Development direct-ICL validity screen: all worlds/seeds, no method/rank/site selection.
 dnames=list(DOMAINS); direct=[]
 for w in DEV:
  for seed in SEEDS:
   prompts=[];targets=[]
   for pi,di in TRAIN+HELD:
    dom=dnames[di];words=DOMAINS[dom];proc=PROCS[pi];inp=words[3]
    prompts.append('Task mapping:\n'+demos(pi,dom)+f'For {dom}, {proc} {inp} gives')
    tar=words[0] if proc in ('prepend','reverse','rotate') else words[-1]
    targets.append(tok(' '+tar,add_special_tokens=False)['input_ids'][0])
   with torch.no_grad():z=model(**tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False)).logits[:,-1,:]
   direct.append({'world':w,'seed':seed,'accuracy':float((z.argmax(-1)==torch.tensor(targets)).float().mean())})
 if min(r['accuracy'] for r in direct)<=0.125: raise RuntimeError('development direct-ICL validity gate failed; fresh must remain unopened')
 # Fixed task split and sample serialization smoke.
 d=demos(0,'fruit');v=item_vec(model,tok,0,'fruit')
 blob=payload({'layer':LAYER,'vector':v.contiguous(),'pair':[0,0]})
 (ART/'development_smoke.json').write_text(json.dumps({'single_token_count':len(sanity),'multi_token':multi,'train_pairs':TRAIN,'held_pairs':HELD,'hidden_dim':int(v.numel()),'payload_bytes':len(blob),'layer':LAYER,'direct_icl_development':direct},indent=2)+'\n')
 print(json.dumps({'development':'passed','direct_icl':direct,'sample_payload_bytes':len(blob)}))

def make_prompt(p,d):
 return 'Task mapping:\n'+demos(p,d)+query(p,d)

def evaluate_world(model,tok,w,s,devbasis):
 # One evaluation per logical function pair (fixed deterministic construction; world/seed vary term order).
 g=torch.Generator().manual_seed(w*100003+s*7919+5)
 pairs=TRAIN+HELD
 order=torch.randperm(len(pairs),generator=g).tolist();pairs=[pairs[i] for i in order]
 dnames=list(DOMAINS); vectors={pair:item_vec(model,tok,pair[0],dnames[pair[1]]) for pair in pairs}
 targets=[];qprompts=[];iclprompts=[];which=[]
 for pair in pairs:
  p,d=pair; dom=list(DOMAINS)[d]; words=DOMAINS[dom]; # Query uses the fourth input. Each operation is a deterministic unary map over this ordered vocabulary.
  inp=words[3]
  target=words[0] if PROCS[p] in ('prepend','reverse','rotate') else words[-1]
  q=f'For {dom}, {PROCS[p]} {inp} gives'
  qprompts.append('Task mapping:\n'+q);iclprompts.append('Task mapping:\n'+demos(p,dom)+q);targets.append(tok(' '+target,add_special_tokens=False)['input_ids'][0]);which.append(pair)
 t=torch.tensor(targets)
 t0=time.perf_counter();direct=logits(model,tok,iclprompts);directsec=time.perf_counter()-t0
 t0=time.perf_counter();base=logits(model,tok,qprompts);basesec=time.perf_counter()-t0
 def metric(z):
  pr=z.softmax(-1);return float((z.argmax(-1)==t).float().mean()),float(-pr[torch.arange(len(t)),t].log().mean())
 results=[]
 for method,z,elapsed in [('direct_icl',direct,directsec),('query_only',base,basesec)]:
  a,n=metric(z);results.append({'method':method,'accuracy':a,'nll':n,'seconds':elapsed})
 # Evaluate per-task explicit deltas and several codes. Training basis only uses TRAIN IDs from current bank.
 tr=torch.stack([vectors[pair] for pair in pairs if pair in TRAIN]); trmean=tr.mean(0);_,_,vv=torch.linalg.svd(tr-trmean,full_matrices=False);basis=vv[:RANK].T.contiguous()
 heldvec=torch.stack([vectors[pair] for pair in pairs if pair in HELD]);
 # Generic PCA: per task coefficient is a flat rank-R vector.
 # Factorized m: rank-R procedure and domain factors; shared total rank kept equal (R/2 each).
 # Fit each decomposition only on training pair vectors.
 proc_means=[];dom_means=[]
 for i in range(4):
  vals=[vectors[pair] for pair in TRAIN if pair[0]==i];proc_means.append(torch.stack(vals).mean(0))
  vals=[vectors[pair] for pair in TRAIN if pair[1]==i];dom_means.append(torch.stack(vals).mean(0))
 proc_means=torch.stack(proc_means);dom_means=torch.stack(dom_means)
 # Decode factorized hidden-pair estimate with additive two-factor means, subtracting global mean once.
 global_mean=trmean
 fac_basis=basis[:,:2]
 pcoord=(proc_means-global_mean)@fac_basis;dcoord=(dom_means-global_mean)@fac_basis
 recs={};
 for pair in pairs:
  p,d=pair; flat=global_mean+((vectors[pair]-global_mean)@basis)@basis.T
  fac=global_mean+((vectors[pair]-global_mean)@fac_basis)@fac_basis.T
  recs[('generic',pair)]=flat;recs[('factorized',pair)]=fac;recs[('explicit',pair)]=vectors[pair]
 for method in ['explicit','generic','factorized','procedure_only','domain_only']:
  vs=[]
  for pair in pairs:
   if method=='procedure_only':vs.append(proc_means[pair[0]])
   elif method=='domain_only':vs.append(dom_means[pair[1]])
   else:vs.append(recs[(method,pair)] if method in ('explicit','generic','factorized') else vectors[pair])
  t0=time.perf_counter();z=logits(model,tok,qprompts,torch.stack(vs));secs=time.perf_counter()-t0;a,n=metric(z);results.append({'method':method,'accuracy':a,'nll':n,'seconds':secs})
 blobs=[]
 # Charge full table for each bank, including bases and labels.
 explicit=payload({'method':'explicit','pairs':pairs,'vectors':torch.stack([vectors[x] for x in pairs]).contiguous()})
 generic=payload({'method':'generic_pca','pairs':pairs,'mean':global_mean.contiguous(),'basis':basis.contiguous(),'coords':torch.stack([(vectors[x]-global_mean)@basis for x in pairs]).half().contiguous(),'rank':RANK})
 factor=payload({'method':'factorized','pairs':pairs,'global_mean':global_mean.contiguous(),'basis':fac_basis.contiguous(),'procedure_coords':pcoord.half().contiguous(),'domain_coords':dcoord.half().contiguous(),'task_coords':torch.stack([(vectors[x]-global_mean)@fac_basis for x in pairs]).half().contiguous(),'rank_per_factor':2})
 return results,[('explicit',explicit),('generic',generic),('factorized',factor)],vectors

def fresh():
 model,tok=load(); assert not [k for k,v in token_sanity(tok).items() if len(v)!=1]
 train=DEV
 bank=[]
 for w in FRESH:
  for s in SEEDS:
   out=evaluate_world(model,tok,w,s,None);bank.append((w,s,*out))
 rows=[]
 base_dir=Path(snapshot_download(MODEL,revision=REV,tqdm_class=None)); names=['model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json'];basebytes=sum((base_dir/n).stat().st_size for n in names if (base_dir/n).is_file())
 for w,s,metrics,blobs,vecs in bank:
  for method,b in blobs:
   path=PAY/f'{w}_{s}_{method}.pt';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);m=next(x for x in metrics if x['method']==method)
   rows.append({'world':w,'seed':s,'method':method,'accuracy':m['accuracy'],'mean_target_nll':m['nll'],'inference_seconds':m['seconds'],'payload_bytes':len(b),'base_model_bytes':basebytes,'total_deployment_bytes':len(b)+basebytes,'hash':hashlib.sha256(b).hexdigest(),'path':str(path.relative_to(ROOT.parents[2]))})
  for m in metrics:
   if m['method'] in ('direct_icl','query_only'):rows.append({'world':w,'seed':s,'method':m['method'],'accuracy':m['accuracy'],'mean_target_nll':m['nll'],'inference_seconds':m['seconds'],'payload_bytes':0,'base_model_bytes':basebytes,'total_deployment_bytes':basebytes,'hash':'','path':''})
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 (ART/'fresh_runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps({'rows':len(rows),'base_model_bytes':basebytes}))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['development','fresh'],required=True);a=p.parse_args()
 dev() if a.phase=='development' else fresh()
