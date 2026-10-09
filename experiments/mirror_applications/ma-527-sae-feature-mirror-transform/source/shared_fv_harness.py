#!/usr/bin/env python3
"""Extract and compress support-derived function vectors using pinned Pythia-70M."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
TASKS=[
 ('country_capital',[('France','Paris'),('Italy','Rome'),('Spain','Madrid'),('Germany','Berlin'),('Japan','Tokyo'),('Canada','Ottawa'),('Brazil','Brasilia'),('Egypt','Cairo'),('India','New Delhi'),('Norway','Oslo'),('Peru','Lima'),('Greece','Athens'),('Kenya','Nairobi'),('Poland','Warsaw'),('Chile','Santiago'),('Thailand','Bangkok')]),
 ('capital_country',[('Paris','France'),('Rome','Italy'),('Madrid','Spain'),('Berlin','Germany'),('Tokyo','Japan'),('Ottawa','Canada'),('Brasilia','Brazil'),('Cairo','Egypt'),('New Delhi','India'),('Oslo','Norway'),('Lima','Peru'),('Athens','Greece'),('Nairobi','Kenya'),('Warsaw','Poland'),('Santiago','Chile'),('Bangkok','Thailand')]),
 ('country_currency',[('France','euro'),('Japan','yen'),('Canada','Canadian dollar'),('Brazil','real'),('Egypt','Egyptian pound'),('India','rupee'),('Norway','krone'),('Peru','sol'),('Poland','zloty'),('Thailand','baht'),('Kenya','shilling'),('Switzerland','franc'),('Turkey','lira'),('Russia','ruble'),('South Korea','won'),('China','yuan')]),
 ('currency_country',[('euro','France'),('yen','Japan'),('Canadian dollar','Canada'),('real','Brazil'),('Egyptian pound','Egypt'),('rupee','India'),('krone','Norway'),('sol','Peru'),('zloty','Poland'),('baht','Thailand'),('shilling','Kenya'),('franc','Switzerland'),('lira','Turkey'),('ruble','Russia'),('won','South Korea'),('yuan','China')]),
 ('state_postal',[('California','CA'),('Washington','WA'),('Oregon','OR'),('Nevada','NV'),('Arizona','AZ'),('Utah','UT'),('Colorado','CO'),('New Mexico','NM'),('Texas','TX'),('Oklahoma','OK'),('Kansas','KS'),('Nebraska','NE'),('South Dakota','SD'),('North Dakota','ND'),('Minnesota','MN'),('Iowa','IA')]),
 ('postal_state',[('CA','California'),('WA','Washington'),('OR','Oregon'),('NV','Nevada'),('AZ','Arizona'),('UT','Utah'),('CO','Colorado'),('NM','New Mexico'),('TX','Texas'),('OK','Oklahoma'),('KS','Kansas'),('NE','Nebraska'),('SD','South Dakota'),('ND','North Dakota'),('MN','Minnesota'),('IA','Iowa')]),
 ('element_symbol',[('Hydrogen','H'),('Helium','He'),('Lithium','Li'),('Carbon','C'),('Nitrogen','N'),('Oxygen','O'),('Fluorine','F'),('Neon','Ne'),('Sodium','Na'),('Magnesium','Mg'),('Aluminum','Al'),('Silicon','Si'),('Phosphorus','P'),('Sulfur','S'),('Chlorine','Cl'),('Argon','Ar')]),
 ('symbol_element',[('H','Hydrogen'),('He','Helium'),('Li','Lithium'),('C','Carbon'),('N','Nitrogen'),('O','Oxygen'),('F','Fluorine'),('Ne','Neon'),('Na','Sodium'),('Mg','Magnesium'),('Al','Aluminum'),('Si','Silicon'),('P','Phosphorus'),('S','Sulfur'),('Cl','Chlorine'),('Ar','Argon')]),
 ('number_digit',[('one','1'),('two','2'),('three','3'),('four','4'),('five','5'),('six','6'),('seven','7'),('eight','8'),('nine','9'),('ten','10'),('eleven','11'),('twelve','12'),('thirteen','13'),('fourteen','14'),('fifteen','15'),('sixteen','16')]),
 ('digit_number',[('1','one'),('2','two'),('3','three'),('4','four'),('5','five'),('6','six'),('7','seven'),('8','eight'),('9','nine'),('10','ten'),('11','eleven'),('12','twelve'),('13','thirteen'),('14','fourteen'),('15','fifteen'),('16','sixteen')]),
 ('animal_sound',[('dog','bark'),('cat','meow'),('cow','moo'),('sheep','baa'),('horse','neigh'),('duck','quack'),('lion','roar'),('snake','hiss'),('frog','croak'),('owl','hoot'),('bee','buzz'),('pig','oink'),('rooster','crow'),('wolf','howl'),('elephant','trumpet'),('donkey','bray')]),
 ('sound_animal',[('bark','dog'),('meow','cat'),('moo','cow'),('baa','sheep'),('neigh','horse'),('quack','duck'),('roar','lion'),('hiss','snake'),('croak','frog'),('hoot','owl'),('buzz','bee'),('oink','pig'),('crow','rooster'),('howl','wolf'),('trumpet','elephant'),('bray','donkey')]),
 ('english_french',[('cat','chat'),('dog','chien'),('house','maison'),('water','eau'),('red','rouge'),('blue','bleu'),('book','livre'),('apple','pomme'),('bread','pain'),('sun','soleil'),('moon','lune'),('tree','arbre'),('car','voiture'),('friend','ami'),('school','ecole'),('small','petit')]),
 ('french_english',[('chat','cat'),('chien','dog'),('maison','house'),('eau','water'),('rouge','red'),('bleu','blue'),('livre','book'),('pomme','apple'),('pain','bread'),('soleil','sun'),('lune','moon'),('arbre','tree'),('voiture','car'),('ami','friend'),('ecole','school'),('petit','small')]),
 ('english_spanish',[('cat','gato'),('dog','perro'),('house','casa'),('water','agua'),('red','rojo'),('blue','azul'),('book','libro'),('apple','manzana'),('bread','pan'),('sun','sol'),('moon','luna'),('tree','arbol'),('car','coche'),('friend','amigo'),('school','escuela'),('small','pequeno')]),
 ('spanish_english',[('gato','cat'),('perro','dog'),('casa','house'),('agua','water'),('rojo','red'),('azul','blue'),('libro','book'),('manzana','apple'),('pan','bread'),('sol','sun'),('luna','moon'),('arbol','tree'),('coche','car'),('amigo','friend'),('escuela','school'),('pequeno','small')])
]
D=512; BASIS_TASKS=tuple(range(12)); HELD_TASKS=tuple(range(12,16)); RANKS=(1,2,4,8,12)
MODEL_SHA='3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd'
CONFIG_SHA='002050231a9b1ec3ac77aa6b9b3bbdc4d923f4068a7dd33b8da72a9bd6ad9a43'
TOKENIZER_SHA='c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624'
REVISION='e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'
LAYER_OUT=4; HOOK_LAYER=3

def split_task(task_id,seed):
 rng=np.random.default_rng(seed+1009*task_id)
 ids=rng.permutation(16)
 return [TASKS[task_id][1][i] for i in ids[:8]],[TASKS[task_id][1][i] for i in ids[8:]]

def load_model(model_dir):
 from transformers import AutoModelForCausalLM,AutoTokenizer
 model_dir=Path(model_dir)
 w=model_dir/'model.safetensors'
 if hashlib.sha256(w.read_bytes()).hexdigest()!=MODEL_SHA: raise ValueError('pinned model hash mismatch')
 if hashlib.sha256((model_dir/'config.json').read_bytes()).hexdigest()!=CONFIG_SHA: raise ValueError('pinned config hash mismatch')
 if hashlib.sha256((model_dir/'tokenizer.json').read_bytes()).hexdigest()!=TOKENIZER_SHA: raise ValueError('pinned tokenizer hash mismatch')
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=None)
 model.eval()
 import torch
 torch.set_num_threads(5)
 return model,tok,torch

def enc(tok,s): return tok.encode(s,add_special_tokens=False)

def prompt(query,demos=()):
 lines=[f'Input: {x}\nOutput: {y}' for x,y in demos]
 lines.append(f'Input: {query}\nOutput:')
 return '\n'.join(lines)

def get_activation(model,tok,torch,text):
 ids=enc(tok,text)
 x=torch.tensor([ids],dtype=torch.long)
 with torch.no_grad(): out=model(input_ids=x,output_hidden_states=True,use_cache=False)
 return out.hidden_states[LAYER_OUT][0,-1].detach().cpu().numpy().astype(np.float32),len(ids)

def extract_task_vectors(model,tok,torch,seed):
 vectors=[]; manifest=[]; all_support_tokens=0
 for tid,(name,pairs) in enumerate(TASKS):
  support,queries=split_task(tid,seed)
  ds=[]; token_count=0
  for q in support:
   demos=[x for x in support if x[0]!=q[0]]
   full,full_tokens=get_activation(model,tok,torch,prompt(q[0],demos))
   base,base_tokens=get_activation(model,tok,torch,prompt(q[0]))
   token_count+=full_tokens+base_tokens
   ds.append(full-base)
  vectors.append(np.mean(ds,axis=0,dtype=np.float64).astype(np.float32))
  manifest.append(dict(task_id=tid,name=name,support=support,evaluation=queries))
  all_support_tokens+=token_count
 return np.stack(vectors),manifest,dict(forward_calls=16*8*2,input_tokens=all_support_tokens)

def hook_forward(model,torch,ids,mask,pos,vec):
 layer=model.gpt_neox.layers[HOOK_LAYER]
 handle=None
 if vec is not None:
  v=torch.as_tensor(vec,dtype=torch.float32)
  def add(module,args,out):
   h=out[0].clone();h[:,pos,:]+=v
   return (h,)+tuple(out[1:])
  handle=layer.register_forward_hook(add)
 try:
  with torch.no_grad(): out=model(input_ids=ids,attention_mask=mask,use_cache=False)
  return out.logits
 finally:
  if handle is not None: handle.remove()

def score_candidates(model,tok,torch,query,candidates,vec):
 prefix=prompt(query)
 pids=enc(tok,prefix); seqs=[]; targets=[]
 for candidate in candidates:
  full=enc(tok,prefix+' '+candidate)
  if full[:len(pids)]!=pids: raise ValueError('candidate tokenization changed the frozen prompt prefix')
  seqs.append(full);targets.append(full[len(pids):])
 maxlen=max(map(len,seqs));pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 ids=torch.full((len(seqs),maxlen),pad,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,s in enumerate(seqs):ids[i,:len(s)]=torch.tensor(s);mask[i,:len(s)]=1
 logits=hook_forward(model,torch,ids,mask,len(pids)-1,vec)
 lp=torch.log_softmax(logits,dim=-1)
 scores=[]
 for i,ans in enumerate(targets):
  sc=0.
  for j,tokid in enumerate(ans):
   sc+=float(lp[i,len(pids)+j-1,tokid].item())
  scores.append(sc)
 return np.asarray(scores,dtype=np.float64),len(seqs),sum(map(len,seqs))

def evaluate(model,tok,torch,vectors,method_vectors,manifest,task_ids=HELD_TASKS):
 bytask=[];total_correct=0;total=0;gold=[];margins=[];forward_calls=0;sequences=0;input_tokens=0
 for tid in task_ids:
  task=manifest[tid];queries=task['evaluation'];candidates=sorted({y for _,y in queries})
  correct=0;gold_task=[];margin_task=[]
  for x,y in queries:
   sc,ncalls,ntokens=score_candidates(model,tok,torch,x,candidates,method_vectors[tid]);forward_calls+=ncalls;sequences+=len(candidates);input_tokens+=ntokens
   gi=candidates.index(y);correct+=int(int(np.argmax(sc))==gi);gold_task.append(float(sc[gi]))
   wrong=np.delete(sc,gi);margin_task.append(float(sc[gi]-np.max(wrong)))
  bytask.append(dict(task_id=tid,name=task['name'],accuracy=correct/len(queries),mean_gold_logprob=float(np.mean(gold_task)),mean_gold_margin=float(np.mean(margin_task))))
  total_correct+=correct;total+=len(queries);gold.extend(gold_task);margins.extend(margin_task)
 return dict(heldout_accuracy=total_correct/total,mean_gold_candidate_logprob=float(np.mean(gold)),mean_gold_vs_best_negative_margin=float(np.mean(margins)),queries=total,candidate_forward_calls=forward_calls,candidate_sequences=sequences,candidate_input_tokens=input_tokens,by_task=bytask)

def pca_fit(vecs,rank):
 train=vecs[list(BASIS_TASKS)].astype(np.float64)
 mean=train.mean(axis=0).astype(np.float32)
 _,_,vt=np.linalg.svd(train-mean,full_matrices=False)
 basis=vt[:rank].T.astype(np.float32)
 codes=(vecs-mean)@basis
 decoded=mean+codes@basis.T
 return mean,basis,codes,decoded.astype(np.float32)


def representation_kind(label,param):
 if label=='none': return 0,0
 if label=='explicit_fv': return 1,0
 return 2,abs(param)

def save_payload(path,kind,vecs,rank=0,parts=None):
 meta=dict(task_ids=np.arange(16,dtype=np.int16),layer=np.array([LAYER_OUT],np.int16),
           model_revision=np.frombuffer(REVISION.encode(),dtype='S40'),model_sha256=np.frombuffer(MODEL_SHA.encode(),dtype='S64'),
           schema=np.array([516,kind,rank],np.int32))
 if kind==0: return 0
 if kind==1: arrays=dict(function_vectors=vecs,**meta)
 else:
  mean,basis,codes=parts
  arrays=dict(shared_mean=mean,shared_basis=basis,function_codes=codes,**meta)
 np.savez(path,**arrays)
 return Path(path).stat().st_size

def run(seed,model_dir,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 model,tok,torch=load_model(model_dir)
 t0=time.perf_counter();vecs,manifest,support_compute=extract_task_vectors(model,tok,torch,seed);extract_s=time.perf_counter()-t0
 common_files=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
 common_base_bytes=sum((Path(model_dir)/name).stat().st_size for name in common_files)
 (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 np.savez(out/'extracted_fvs.npz',task_vectors=vecs,task_ids=np.arange(16,dtype=np.int16))
 rows=[]
 methods=[('none',0),('explicit_fv',1)]+[(f'mirror_r{r}',r) for r in RANKS]+[(f'native_pca_r{r}',-r) for r in RANKS]
 for label,param in methods:
  kind,rank=representation_kind(label,param)
  if kind==0: decoded=np.zeros_like(vecs);parts=None;rank_payload=0
  elif kind==1: decoded=vecs.copy();parts=None;rank_payload=0
  else:
   mean,basis,codes,decoded=pca_fit(vecs,rank);parts=(mean,basis,codes);rank_payload=rank
  path=out/(label+'.npz')
  nbytes=save_payload(path,kind,vecs,rank_payload,parts) if kind else 0
  start=time.perf_counter();metrics=evaluate(model,tok,torch,vecs,decoded,manifest);infer_s=time.perf_counter()-start
  if label.startswith('native_pca_'):
   mirror_label='mirror_r'+str(rank);mirror=next(r for r in rows if r['method']==mirror_label)
   alias=nbytes==mirror['payload_bytes'] and metrics==mirror['metrics']
  else:alias=False
  extra=(512*256 + 4*512*rank) if rank else (512*256 if kind==1 else 0)
  rows.append(dict(seed=seed,method=label,rank=rank,payload_bytes=nbytes,
                   total_deployment_bytes=common_base_bytes+nbytes,common_base_bytes=common_base_bytes,extra_compute_proxy=extra,
                   extraction_seconds=extract_s,inference_seconds=infer_s,support_examples=16*8,
                   support_forward_calls=support_compute['forward_calls'],support_input_tokens=support_compute['input_tokens'],
                   evaluation_queries=32,metrics=metrics,native_alias=alias))
 report=dict(experiment_id='MA-516',seed=seed,model='EleutherAI/pythia-70m-deduped',revision=REVISION,
             model_sha256=MODEL_SHA,dimension=512,extraction_seconds=extract_s,rows=rows)
 (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'seed':seed,'extraction_seconds':extract_s,'rows':len(rows),'heldout_rank4':[r for r in rows if r['method'] in ('none','explicit_fv','mirror_r4','native_pca_r4')]},ensure_ascii=False,indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
