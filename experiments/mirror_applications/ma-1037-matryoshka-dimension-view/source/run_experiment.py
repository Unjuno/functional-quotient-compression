#!/usr/bin/env python3
"""MA-1037. Nomic Matryoshka embeddings + SciFact exact retrieval audit."""
import argparse, csv, hashlib, json, math, os, statistics, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import onnxruntime as ort
from transformers import AutoTokenizer

SRC=Path(__file__).resolve().parent
ROOT=SRC.parent
D=768; DIMS=(64,128,256,512,768); GRID=np.linspace(-math.pi/2,math.pi/2,21)
MODEL_REV='e9b6763023c676ca8431644204f50c2b100d9aab'
CODE_REV='7710840340a098cfb869c4f65e87cf2b1b70caca'
DATA_REV='b3b5335604bf5ee3c4447671af975ea25143d4f5'
QRELS_REV='2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029'

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()

def read_data(data_dir):
 data_dir=Path(data_dir)
 ct=pq.read_table(data_dir/'corpus.parquet').to_pylist()
 qt=pq.read_table(data_dir/'queries.parquet').to_pylist()
 corpus_ids=[str(x['_id']) for x in ct]
 corpus_texts=['search_document: '+str(x['title']).strip()+' '+str(x['text']).strip() for x in ct]
 query_ids=[str(x['_id']) for x in qt]
 query_texts=['search_query: '+str(x['text']).strip() for x in qt]
 return corpus_ids,corpus_texts,query_ids,query_texts

def encode_texts(session,tokenizer,texts,batch_size,max_length):
 outs=[]; t0=time.perf_counter()
 names={x.name for x in session.get_inputs()}
 for lo in range(0,len(texts),batch_size):
  batch=tokenizer(texts[lo:lo+batch_size],padding=True,truncation=True,max_length=max_length,return_tensors='np')
  feeds={name:np.asarray(batch[name],dtype=np.int64) for name in names}
  hidden=session.run(['last_hidden_state'],feeds)[0]
  mask=np.asarray(batch['attention_mask'],dtype=np.float32)[:,:,None]
  pooled=(hidden*mask).sum(1)/np.maximum(mask.sum(1),1.)
  mean=pooled.mean(axis=1,keepdims=True); var=np.mean((pooled-mean)**2,axis=1,keepdims=True)
  pooled=(pooled-mean)/np.sqrt(var+1e-5)
  pooled=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)
  outs.append(pooled.astype(np.float32,copy=False))
 wall=time.perf_counter()-t0
 return np.ascontiguousarray(np.concatenate(outs,axis=0),dtype=np.float32),wall

def load_embeddings():
 x=np.load(ROOT/'runs'/'corpus_vectors.npy',allow_pickle=False)
 q=np.load(ROOT/'runs'/'query_vectors.npy',allow_pickle=False)
 ids=json.loads((ROOT/'runs'/'embedding_ids.json').read_text())
 assert x.shape==(len(ids['corpus_ids']),D) and q.shape==(len(ids['query_ids']),D)
 return x,q,ids

def normalize(x):
 n=np.linalg.norm(x,axis=1,keepdims=True)
 return np.asarray(x/np.maximum(n,1e-12),dtype=np.float32)

def prefix(x,d):
 return np.ascontiguousarray(normalize(x[:,:d]))

def givens_prefix(x,d,theta):
 if d==D: return prefix(x,d)
 c=np.float32(math.cos(float(theta))); s=np.float32(math.sin(float(theta)))
 z=np.array(x,copy=True)
 a=z[:,d-1].copy(); b=z[:,d].copy()
 z[:,d-1]=c*a-s*b; z[:,d]=s*a+c*b
 return np.ascontiguousarray(normalize(z[:,:d]))

def linear_boundary_prefix(x,d,weights):
 if d==D: return prefix(x,d)
 w=np.asarray(weights,dtype=np.float32); w=w/(np.linalg.norm(w)+1e-12)
 z=np.array(x[:,:d],copy=True)
 z[:,d-1]=w[0]*x[:,d-1]+w[1]*x[:,d]
 return np.ascontiguousarray(normalize(z))

def qrels(path):
 out={}
 with open(path,newline='') as f:
  for r in csv.DictReader(f,delimiter='\t'):
   q=str(r['query-id']); doc=str(r['corpus-id']); rel=int(r['score'])
   if rel>0: out.setdefault(q,{})[doc]=rel
 return out

def exact_topk(query_vectors,doc_vectors,k=10):
 scores=query_vectors@doc_vectors.T
 kk=min(k,doc_vectors.shape[0]); part=np.argpartition(-scores,kk-1,axis=1)[:,:kk]
 rr=np.arange(len(query_vectors))[:,None]; ss=scores[rr,part]
 order=np.argsort(-ss,axis=1,kind='stable')
 return part[rr,order]

def ranking_metrics(query_vectors,doc_vectors,query_ids,doc_ids,rel_map,k=10):
 top=exact_topk(query_vectors,doc_vectors,k)
 vals=[]
 for i,qid in enumerate(query_ids):
  rel=rel_map.get(qid,{})
  got=[doc_ids[j] for j in top[i]]
  gains=np.asarray([2**rel.get(doc,0)-1 for doc in got],dtype=np.float64)
  discounts=1/np.log2(np.arange(2,len(got)+2))
  dcg=float(np.sum(gains*discounts))
  ideal=sorted((2**r-1 for r in rel.values()),reverse=True)[:k]
  idcg=float(np.sum(np.asarray(ideal)*1/np.log2(np.arange(2,len(ideal)+2)))) if ideal else 0.
  ndcg=dcg/idcg if idcg else 0.
  relevant={doc for doc,r in rel.items() if r>0}
  hits=[rank+1 for rank,doc in enumerate(got) if doc in relevant]
  recall=len(hits)/len(relevant) if relevant else 0.
  mrr=1/hits[0] if hits else 0.
  vals.append((ndcg,recall,mrr))
 a=np.asarray(vals,dtype=np.float64)
 return {'nDCG@10':float(a[:,0].mean()),'Recall@10':float(a[:,1].mean()),'MRR@10':float(a[:,2].mean())}

def qid_subset(rel,query_map):
 ids=sorted(set(rel)&set(query_map)); idx=np.asarray([query_map[q] for q in ids],dtype=np.int64)
 return ids,idx

def dev_select():
 corpus,queries,ids=load_embeddings(); qi={q:i for i,q in enumerate(ids['query_ids'])}
 train=qrels(Path(os.environ.get('MA1037_DATA_DIR','/tmp/ma1037-scifact'))/'train.tsv')
 train_ids,ix=qid_subset(train,qi); q=queries[ix]; docids=ids['corpus_ids']
 selected={}; grid_scores={}
 for d in DIMS:
  if d==D: selected[str(d)]=0.; grid_scores[str(d)]=[]; continue
  vals=[]
  for theta in GRID:
   Q=givens_prefix(q,d,theta); X=givens_prefix(corpus,d,theta)
   vals.append(ranking_metrics(Q,X,train_ids,docids,train)['nDCG@10'])
  best=int(np.argmax(vals)); selected[str(d)]=float(GRID[best]); grid_scores[str(d)]=[float(v) for v in vals]
 out={'development_qrels':'train.tsv','development_query_count':len(train_ids),'dimension_budgets':list(DIMS),'grid_rad':GRID.tolist(),'angles_rad':selected,'grid_ndcg10':grid_scores,'selection_metric':'mean nDCG@10'}
 (ROOT/'runs'/'DEV_SELECTION.json').write_text(json.dumps(out,indent=2)+'\n')
 return out

def model_bundle_bytes(model_dir):
 model_dir=Path(model_dir)
 files=[model_dir/'onnx'/'model_quantized.onnx',model_dir/'config.json',model_dir/'tokenizer.json',model_dir/'tokenizer_config.json',model_dir/'special_tokens_map.json',model_dir/'vocab.txt']
 return sum(p.stat().st_size for p in files if p.exists()),[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in files if p.exists()]

def save_npy(path,x):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 np.save(path,np.asarray(x,dtype=np.float32),allow_pickle=False)
 return path.stat().st_size,sha256(path)

def retrieval_qps(Q,X,repeats=5):
 _=exact_topk(Q,X,10); times=[]
 for _ in range(repeats):
  t=time.perf_counter(); _=exact_topk(Q,X,10); times.append(time.perf_counter()-t)
 med=float(np.median(times)); return (len(Q)/med if med else float('inf')),med

def audit(model_dir):
 corpus,queries,ids=load_embeddings(); dev=json.loads((ROOT/'runs'/'DEV_SELECTION.json').read_text())
 angles={int(k):float(v) for k,v in dev['angles_rad'].items()}
 data=Path(os.environ.get('MA1037_DATA_DIR','/tmp/ma1037-scifact'))
 test=qrels(data/'test.tsv'); qi={q:i for i,q in enumerate(ids['query_ids'])}; test_ids,ix=qid_subset(test,qi); Qfull=queries[ix]; docids=ids['corpus_ids']
 # Ensure qrels split is truly disjoint at the query-ID level.
 train_ids=set(qrels(data/'train.tsv')); assert not (train_ids&set(test_ids))
 srcpath=ROOT/'payload'/'shared'/'full_source_vectors.npy'; source_bytes,source_hash=save_npy(srcpath,corpus)
 idpath=ROOT/'payload'/'shared'/'corpus_ids.json'; idpath.parent.mkdir(parents=True,exist_ok=True); idpath.write_text(json.dumps(docids,separators=(',',':'))+'\n'); id_bytes=idpath.stat().st_size
 model_bytes,model_files=model_bundle_bytes(model_dir)
 rows=[]; all_metrics={}
 for d in DIMS:
  build_start=time.perf_counter(); Qn=prefix(Qfull,d); Xn=prefix(corpus,d); native_build=time.perf_counter()-build_start
  angles_d=angles[d]
  build_start=time.perf_counter(); Qm=givens_prefix(Qfull,d,angles_d); Xm=givens_prefix(corpus,d,angles_d); mirror_build=time.perf_counter()-build_start
  # The simple normalized 2-coordinate boundary readout has w=(cos(theta), -sin(theta)); it aliases Givens exactly.
  wm=[math.cos(angles_d),-math.sin(angles_d)]
  build_start=time.perf_counter(); Qc=linear_boundary_prefix(Qfull,d,wm); Xc=linear_boundary_prefix(corpus,d,wm); control_build=time.perf_counter()-build_start
  assert np.array_equal(Qm,Qc) and np.array_equal(Xm,Xc)
  for method,Q,X in [('native_prefix',Qn,Xn),('mirror_givens',Qm,Xm),('linear_boundary_control',Qc,Xc)]:
   metrics=ranking_metrics(Q,X,test_ids,docids,test)
   index_path=(ROOT/'payload'/'mirror_givens'/f'candidate_d{d}.npy' if method=='linear_boundary_control' else ROOT/'payload'/method/f'candidate_d{d}.npy')
   if method=='linear_boundary_control':
    candidate_bytes=index_path.stat().st_size; candidate_hash=sha256(index_path)
    assert np.array_equal(np.load(index_path,allow_pickle=False),X)
   else:
    candidate_bytes,candidate_hash=save_npy(index_path,X)
   coordinate_bytes=0 if method=='native_prefix' or d==D else (4 if method=='mirror_givens' else 8)
   coordinate_path=ROOT/'payload'/method/f'coordinate_d{d}.bin'
   if coordinate_bytes:
    if method=='mirror_givens': coordinate_path.write_bytes(np.asarray([angles_d],dtype='<f4').tobytes())
    else: coordinate_path.write_bytes(np.asarray(wm,dtype='<f4').tobytes())
    coordinate_bytes=coordinate_path.stat().st_size
   manifest={'method':method,'dimension':d,'rows':len(docids),'source_embedding':'payload/shared/full_source_vectors.npy','candidate':'candidate_d%d.npy'%d,'coordinate':(angles_d if method=='mirror_givens' else wm) if d<D and method!='native_prefix' else None,'normalized_after_prefix':True,'model_revision':MODEL_REV}
   mp=ROOT/'payload'/method/f'manifest_d{d}.json'; mp.parent.mkdir(parents=True,exist_ok=True); mp.write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n'); mb=mp.stat().st_size
   total=model_bytes+source_bytes+id_bytes+candidate_bytes+coordinate_bytes+mb
   qps,wall=retrieval_qps(Q,X)
   flops=2*len(Q)*len(docids)*d+(6*len(Q) if method=='mirror_givens' and d<D else 0)
   rows.append({'split':'test','method':method,'dimension':d,'serialized_bytes':total,'model_payload_bytes':model_bytes,'source_embedding_bytes':source_bytes,'candidate_index_bytes':candidate_bytes,'coordinate_bytes':coordinate_bytes,'metadata_bytes':id_bytes+mb,'nDCG@10':metrics['nDCG@10'],'Recall@10':metrics['Recall@10'],'MRR@10':metrics['MRR@10'],'retrieval_qps':qps,'retrieval_wall_s':wall,'candidate_build_wall_s':{'native_prefix':native_build,'mirror_givens':mirror_build,'linear_boundary_control':control_build}[method],'active_compute_proxy_flops':int(flops),'candidate_sha256':candidate_hash})
   all_metrics[f'{method}/d{d}']=metrics
 (ROOT/'runs'/'AUDIT_RESULTS.json').write_text(json.dumps({'test_query_count':len(test_ids),'test_qrel_rows':sum(len(v) for v in test.values()),'rows':rows,'metrics':all_metrics,'model_bundle_bytes':model_bytes,'model_bundle_files':model_files,'full_source_vectors_bytes':source_bytes,'full_source_sha256':source_hash},indent=2)+'\n')
 # Per-dimension materialized-system Pareto accounting charges source/model once plus each candidate and all view codes.
 by={(r['method'],r['dimension']):r for r in rows}; bank=[]
 for method in ('native_prefix','mirror_givens','linear_boundary_control'):
  candidates=sum(by[(method,d)]['candidate_index_bytes'] for d in DIMS)
  coords=sum(by[(method,d)]['coordinate_bytes'] for d in DIMS)
  manifests=sum(json.loads((ROOT/'payload'/method/f'manifest_d{d}.json').read_text()) is not None and (ROOT/'payload'/method/f'manifest_d{d}.json').stat().st_size for d in DIMS)
  total=model_bytes+source_bytes+id_bytes+candidates+coords+manifests
  bank.append({'method':method,'complete_five_budget_bank_bytes':int(total),'source_embedding_bytes':source_bytes,'candidate_tables_bytes':candidates,'coordinate_bytes':coords,'model_bundle_bytes':model_bytes})
 (ROOT/'runs'/'BANK_BYTES.json').write_text(json.dumps(bank,indent=2)+'\n')
 return rows

def embed(model_dir,data_dir,batch_size=16,max_length=512):
 corpus_ids,corpus_texts,query_ids,query_texts=read_data(data_dir)
 so=ort.SessionOptions(); so.intra_op_num_threads=int(os.environ.get('MA1037_ONNX_THREADS','4')); so.inter_op_num_threads=1
 session=ort.InferenceSession(str(Path(model_dir)/'onnx'/'model_quantized.onnx'),sess_options=so,providers=['CPUExecutionProvider'])
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
 corpus,ct=encode_texts(session,tok,corpus_texts,batch_size,max_length)
 queries,qt=encode_texts(session,tok,query_texts,batch_size,max_length)
 assert corpus.shape==(5183,D) and queries.shape==(1109,D)
 cp=ROOT/'runs'/'corpus_vectors.npy'; qp=ROOT/'runs'/'query_vectors.npy'
 np.save(cp,corpus,allow_pickle=False); np.save(qp,queries,allow_pickle=False)
 ids={'corpus_ids':corpus_ids,'query_ids':query_ids}
 (ROOT/'runs'/'embedding_ids.json').write_text(json.dumps(ids,separators=(',',':'))+'\n')
 data_dir=Path(data_dir)
 assets=[data_dir/x for x in ('corpus.parquet','queries.parquet','train.tsv','test.tsv')]
 meta={'model_revision':MODEL_REV,'model_weight_sha256':sha256(Path(model_dir)/'onnx'/'model_quantized.onnx'),'model_weight_bytes':(Path(model_dir)/'onnx'/'model_quantized.onnx').stat().st_size,'corpus_embedding_shape':list(corpus.shape),'query_embedding_shape':list(queries.shape),'corpus_encode_wall_s':ct,'query_encode_wall_s':qt,'documents_per_s':len(corpus)/ct,'queries_per_s':len(queries)/qt,'max_length':max_length,'batch_size':batch_size,'onnx_threads':int(os.environ.get('MA1037_ONNX_THREADS','4')),'embedding_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in [cp,qp,ROOT/'runs'/'embedding_ids.json']],'input_files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha256(p)} for p in assets]}
 (ROOT/'runs'/'EMBEDDING_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
 return meta

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['encode','dev','audit'],required=True); ap.add_argument('--model-dir',default=os.environ.get('MA1037_MODEL_DIR','/tmp/ma1037-nomic-model')); ap.add_argument('--data-dir',default=os.environ.get('MA1037_DATA_DIR','/tmp/ma1037-scifact')); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--max-length',type=int,default=512); a=ap.parse_args()
 if a.stage=='encode': print(json.dumps(embed(a.model_dir,a.data_dir,a.batch_size,a.max_length),indent=2))
 elif a.stage=='dev': print(json.dumps(dev_select(),indent=2))
 else: print(json.dumps({'audit_rows':len(audit(a.model_dir))}))
if __name__=='__main__': main()
