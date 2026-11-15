#!/usr/bin/env python3
"""MA-1040 staged run: embed, hash split, dev freeze, then fresh audit."""
import argparse,csv,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import scann
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent.parent
DATA=Path(os.environ.get('MA1040_DATA','/tmp/ma1040-arguana/arguana'))
MODEL=Path(os.environ.get('MA1040_MODEL','/tmp/ma1037-nomic-model'))
D=768
GRID=np.asarray([-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.,0.05,0.10,0.15,0.20,0.25,0.30],dtype=np.float64)
SALT='MA1040|ArguAna|split|'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def jsonl(p):
 with open(p,encoding='utf8') as f:
  for line in f:
   if line.strip(): yield json.loads(line)

def encode_stage():
 corp=list(jsonl(DATA/'corpus.jsonl')); queries=list(jsonl(DATA/'queries.jsonl'))
 did=[str(r['_id']) for r in corp]; qid=[str(r['_id']) for r in queries]
 dt=['search_document: '+str(r.get('title','')).strip()+' '+str(r.get('text','')).strip() for r in corp]
 qt=['search_query: '+str(r.get('text','')).strip() for r in queries]
 opts=ort.SessionOptions(); opts.intra_op_num_threads=int(os.getenv('MA1040_THREADS','4')); opts.inter_op_num_threads=1
 sess=ort.InferenceSession(str(MODEL/'onnx/model_quantized.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
 names={x.name for x in sess.get_inputs()}
 def run(texts,name):
  final=HERE/'runs'/f'{name}.npy'; partial=HERE/'runs'/f'{name}.partial.npy'; progress=HERE/'runs'/f'{name}.progress.json'
  bs=16; maxlen=256; start=0; t0=time.perf_counter()
  if progress.exists() and partial.exists():
   start=int(json.loads(progress.read_text())['next_row'])
  if start==0:
   mm=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(texts),D))
  else:
   mm=np.lib.format.open_memmap(partial,mode='r+',dtype=np.float32,shape=(len(texts),D))
  for i in range(start,len(texts),bs):
   b=tok(texts[i:i+bs],padding=True,truncation=True,max_length=maxlen,return_tensors='np')
   f={n:np.asarray(b[n],dtype=np.int64) for n in names}
   h=sess.run(['last_hidden_state'],f)[0]
   mask=np.asarray(b['attention_mask'],dtype=np.float32)[:,:,None]
   v=(h*mask).sum(1)/np.maximum(mask.sum(1),1.)
   mean=v.mean(axis=1,keepdims=True); var=np.mean((v-mean)**2,axis=1,keepdims=True)
   v=(v-mean)/np.sqrt(var+1e-5); v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
   mm[i:i+len(v)]=v.astype(np.float32,copy=False); mm.flush()
   progress.write_text(json.dumps({'next_row':i+len(v),'total_rows':len(texts),'max_length':maxlen})+'\n')
  del mm; partial.replace(final); progress.unlink(missing_ok=True)
  return np.load(final,mmap_mode='r',allow_pickle=False),time.perf_counter()-t0
 cv,ct=run(dt,'corpus_vectors'); qv,qtsec=run(qt,'query_vectors')
 assert cv.shape==(len(did),D) and qv.shape==(len(qid),D)
 (HERE/'runs').mkdir(parents=True,exist_ok=True)
 np.save(HERE/'runs/corpus_vectors.npy',cv,allow_pickle=False); np.save(HERE/'runs/query_vectors.npy',qv,allow_pickle=False)
 (HERE/'runs/ids.json').write_text(json.dumps({'doc_ids':did,'query_ids':qid},separators=(',',':'))+'\n')
 # Token counts used only for role assignment; labels do not influence this step.
 lengths=np.asarray([len(tok.encode(x,add_special_tokens=True,truncation=True,max_length=256)) for x in qt],dtype=np.int32)
 np.save(HERE/'runs/query_lengths.npy',lengths,allow_pickle=False)
 meta={'corpus_rows':len(did),'query_rows':len(qid),'dimension':D,'corpus_embedding_seconds':ct,'query_embedding_seconds':qtsec,'onnx_threads':opts.intra_op_num_threads,'batch_size':16,'max_length':256,'resume_supported':True,'dataset_archive_sha256':sha(Path('/tmp/ma1040_arguana.zip')),'dataset_asset_sha256':{n:sha(DATA/n) for n in ('corpus.jsonl','queries.jsonl')},'embedding_hashes':{n:sha(HERE/'runs'/n) for n in ('corpus_vectors.npy','query_vectors.npy','ids.json','query_lengths.npy')}}
 (HERE/'runs/EMBEDDING_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps(meta,indent=2))

def role_split():
 ids=json.loads((HERE/'runs/ids.json').read_text())['query_ids']
 lengths=np.load(HERE/'runs/query_lengths.npy',allow_pickle=False)
 dev=[]; audit=[]
 for q in ids:
  byte=hashlib.sha256((SALT+q).encode()).digest()[0]
  (dev if byte<128 else audit).append(q)
 threshold=int(np.median([lengths[ids.index(q)] for q in dev]))
 out={'salt':SALT,'split_rule':'sha256 first byte <128 -> dev, else audit','dev_qids':sorted(dev),'audit_qids':sorted(audit),'dev_qids_sha256':hashlib.sha256('\n'.join(sorted(dev)).encode()).hexdigest(),'audit_qids_sha256':hashlib.sha256('\n'.join(sorted(audit)).encode()).hexdigest(),'role_threshold_wordpieces':threshold,'role_rule':'token_count <= threshold short; greater long','dev_query_count':len(dev),'audit_query_count':len(audit)}
 (HERE/'runs/SPLIT_LOCK.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if 'qids' not in k},indent=2))

def qrels_for(qidset):
 out={}
 with open(DATA/'qrels/test.tsv',newline='',encoding='utf8') as f:
  for r in csv.DictReader(f,delimiter='\t'):
   q=str(r['query-id'])
   if q in qidset and int(r['score'])>0: out.setdefault(q,{})[str(r['corpus-id'])]=int(r['score'])
 return out

def ids_and_arrays(qids):
 ids=json.loads((HERE/'runs/ids.json').read_text()); qi={q:i for i,q in enumerate(ids['query_ids'])}
 ix=np.asarray([qi[q] for q in qids],dtype=np.int64)
 return ids,np.load(HERE/'runs/query_vectors.npy',allow_pickle=False)[ix],np.load(HERE/'runs/query_lengths.npy',allow_pickle=False)[ix]

def metric(top_ids, qids, rel):
 vals=[]
 for q,got in zip(qids,top_ids):
  rr=rel.get(q,{})
  gains=np.asarray([2**rr.get(d,0)-1 for d in got],dtype=np.float64)
  dcg=float(np.sum(gains/np.log2(np.arange(2,len(got)+2))))
  ideal=sorted((2**x-1 for x in rr.values()),reverse=True)[:10]
  idcg=float(np.sum(np.asarray(ideal)/np.log2(np.arange(2,len(ideal)+2)))) if ideal else 0.
  ndcg=dcg/idcg if idcg else 0.
  hits=[i+1 for i,d in enumerate(got) if d in rr]
  vals.append((ndcg, float(bool(hits)), 1./hits[0] if hits else 0.))
 a=np.asarray(vals,dtype=np.float64)
 return {'nDCG@10':float(a[:,0].mean()),'Recall@10':float(a[:,1].mean()),'MRR@10':float(a[:,2].mean())}

def givens(x,theta):
 c=np.float32(math.cos(float(theta))); s=np.float32(math.sin(float(theta)))
 z=np.array(x,copy=True); a=z[:,0].copy(); b=z[:,1].copy()
 z[:,0]=c*a-s*b; z[:,1]=s*a+c*b
 return np.ascontiguousarray(z)

def linear(x,theta):
 # Ordinary full 2x2 matrix readout with the same four matrix coefficients.
 c=np.float32(math.cos(float(theta))); s=np.float32(math.sin(float(theta)))
 z=np.array(x,copy=True); a=z[:,0].copy(); b=z[:,1].copy()
 z[:,0]=c*a-s*b; z[:,1]=s*a+c*b
 return np.ascontiguousarray(z)

def build_index():
 x=np.load(HERE/'runs/corpus_vectors.npy',allow_pickle=False)
 t=time.perf_counter()
 builder=scann.scann_ops_pybind.builder(x,100,'dot_product')
 searcher=(builder.tree(128,16,training_sample_size=min(100000,len(x)),training_iterations=10,random_init=False,avq=1.0)
  .score_ah(2,anisotropic_quantization_threshold=0.2,training_sample_size=min(100000,len(x)),training_iterations=10)
  .reorder(100).set_n_training_threads(4).build())
 wall=time.perf_counter()-t
 target=HERE/'payload/scann_index'; target.mkdir(parents=True,exist_ok=True); searcher.serialize(str(target))
 (HERE/'runs/INDEX_BUILD.json').write_text(json.dumps({'scann_version':'1.4.2','rows':len(x),'dimension':x.shape[1],'build_wall_seconds':wall,'num_leaves':128,'leaves_to_search':16,'avq':1.0,'dimensions_per_block':2,'anisotropic_quantization_threshold':0.2,'reorder_neighbors':100},indent=2)+'\n')
 print(json.dumps({'build_wall_seconds':wall,'payload_files':[str(p.relative_to(HERE)) for p in target.rglob('*') if p.is_file()]},indent=2))

def select_dev():
 split=json.loads((HERE/'runs/SPLIT_LOCK.json').read_text()); qids=sorted(split['dev_qids']); ids,Q,lens=ids_and_arrays(qids); x=np.load(HERE/'runs/corpus_vectors.npy',allow_pickle=False)
 rel=qrels_for(set(qids)); qids=[q for q in qids if q in rel]; ix=np.asarray([ids['query_ids'].index(q) for q in qids]); Q=Q[ix]; lens=lens[ix]
 # Fit a single official ScaNN AVQ searcher; all transformed-query arms share it.
 searcher=scann.scann_ops_pybind.load_searcher(str(HERE/'payload/scann_index'))
 threshold=split['role_threshold_wordpieces']; roles=np.where(lens<=threshold,'short','long')
 grid={}; angles={}
 for role in ('short','long'):
  ix=np.flatnonzero(roles==role); rg=[qids[j] for j in ix]; rrel={q:rel[q] for q in rg}
  scores=[]
  for angle in GRID:
   qs=givens(Q[ix],angle); hits=searcher.search_batched(qs,final_num_neighbors=10,pre_reorder_num_neighbors=100,leaves_to_search=16)[0]
   got=[[ids['doc_ids'][int(j)] for j in row] for row in hits]
   scores.append(metric(got,rg,rrel)['nDCG@10'])
  best=sorted(range(len(GRID)),key=lambda i:(-scores[i],abs(float(GRID[i])),float(GRID[i])))[0]
  angles[role]=float(GRID[best]); grid[role]=scores
 lock={'angles_rad':angles,'grid_rad':GRID.tolist(),'dev_ndcg10_grid':grid,'dev_queries_with_qrels':len(qids),'role_threshold_wordpieces':threshold,'development_qids_sha256':split['dev_qids_sha256'],'audit_opened':False,'optimizer_updates':0,'freeze_claim':'no metric parameters changed after this lock'}
 (HERE/'runs/DEV_FREEZE.json').write_text(json.dumps(lock,indent=2)+'\n')
 print(json.dumps(lock,indent=2))

def audit():
 freeze=json.loads((HERE/'runs/DEV_FREEZE.json').read_text()); split=json.loads((HERE/'runs/SPLIT_LOCK.json').read_text()); threshold=freeze['role_threshold_wordpieces']; angles=freeze['angles_rad']
 qids=sorted(split['audit_qids']); ids,Q,lens=ids_and_arrays(qids); x=np.load(HERE/'runs/corpus_vectors.npy',allow_pickle=False)
 rel=qrels_for(set(qids)); qids=[q for q in qids if q in rel]; ix=np.asarray([ids['query_ids'].index(q) for q in qids]); Q=Q[ix]; lens=lens[ix]; roles=np.where(lens<=threshold,'short','long')
 searcher=scann.scann_ops_pybind.load_searcher(str(HERE/'payload/scann_index'))
 exact=Q@x.T; exact_top=np.argpartition(-exact,9,axis=1)[:,:10]; rr=np.arange(len(Q))[:,None]; es=exact[rr,exact_top]; exact_top=exact_top[rr,np.argsort(-es,axis=1,kind='stable')]
 arms={'native':Q,'mirror':np.vstack([givens(Q[i:i+1],angles[roles[i]]) for i in range(len(Q))]),'linear_control':np.vstack([linear(Q[i:i+1],angles[roles[i]]) for i in range(len(Q))])}
 rows=[]; raw={}
 for name,qs in arms.items():
  hits=searcher.search_batched(qs,final_num_neighbors=10,pre_reorder_num_neighbors=100,leaves_to_search=16)[0]
  got=[[ids['doc_ids'][int(j)] for j in row] for row in hits]
  metrics=metric(got,qids,rel)
  exactrec=float(np.mean([len(set(map(int,hits[i]))&set(map(int,exact_top[i])))/10 for i in range(len(Q))]))
  rows.append({'arm':name,**metrics,'ann_overlap_recall10_vs_exact':exactrec,'queries':len(Q)})
  raw[name]=hits.astype(np.int32).tolist()
 # ScaNN results should exactly alias because Mirror and matched linear transform use the same matrix expression.
 assert np.array_equal(arms['mirror'],arms['linear_control'])
 assert np.array_equal(np.asarray(raw['mirror']),np.asarray(raw['linear_control']))
 bytefiles=[]
 for p in sorted((HERE/'payload/scann_index').rglob('*')):
  if p.is_file(): bytefiles.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
 model_files=['onnx/model_quantized.onnx','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']
 model_bytes=sum((MODEL/f).stat().st_size for f in model_files if (MODEL/f).exists())
 index_bytes=sum(x['bytes'] for x in bytefiles)
 id_path=HERE/'payload/doc_ids.json'; id_path.write_text(json.dumps(ids['doc_ids'],separators=(',',':'))+'\n')
 view_path=HERE/'payload/view_config.json'; view_path.write_text(json.dumps({'role_threshold_wordpieces':threshold,'angles_rad':angles,'rotation_dims':[0,1]},sort_keys=True,separators=(',',':'))+'\n')
 control_path=HERE/'payload/linear_control.json'; control_path.write_text(json.dumps({'role_threshold_wordpieces':threshold,'matrices':{r:[[float(np.float32(math.cos(t))),float(np.float32(-math.sin(t)))],[float(np.float32(math.sin(t))),float(np.float32(math.cos(t)))]] for r,t in angles.items()},'rotation_dims':[0,1]},sort_keys=True,separators=(',',':'))+'\n')
 id_bytes=id_path.stat().st_size; view_bytes=view_path.stat().st_size; control_bytes=control_path.stat().st_size
 common={'model_bytes':model_bytes,'index_bytes':index_bytes,'index_files':bytefiles,'doc_id_map_bytes':id_bytes,'mirror_code_and_role_metadata_bytes':view_bytes,'linear_control_and_role_metadata_bytes':control_bytes,'mirror_full_payload_bytes':model_bytes+index_bytes+id_bytes+view_bytes,'linear_control_full_payload_bytes':model_bytes+index_bytes+id_bytes+control_bytes,'native_full_payload_bytes':model_bytes+index_bytes+id_bytes}
 out={'rows':rows,'payload':common,'audit_qids_sha256':split['audit_qids_sha256'],'audit_qrels_count':len(qids),'qrels_file_sha256':sha(DATA/'qrels/test.tsv'),'audit_opened_after_dev_freeze':True,'matched_control_exact_arrays':True,'searcher_config':json.loads((HERE/'runs/INDEX_BUILD.json').read_text())}
 (HERE/'runs/AUDIT_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
 (HERE/'runs/audit_top10.json').write_text(json.dumps({'qids':qids,'top10':raw},separators=(',',':'))+'\n')
 print(json.dumps(out,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('stage',choices=['encode','split','build','dev','audit']); a=p.parse_args()
 {'encode':encode_stage,'split':role_split,'build':build_index,'dev':select_dev,'audit':audit}[a.stage]()
