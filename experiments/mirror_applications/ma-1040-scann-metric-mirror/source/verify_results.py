#!/usr/bin/env python3
"""Independent metric, split, payload and matched-control replay for MA-1040."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from run_experiment import HERE,DATA,MODEL,SALT,GRID,givens,linear
import scann

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def rels_for(qids):
 out={}
 with open(DATA/'qrels/test.tsv',newline='',encoding='utf8') as f:
  for r in csv.DictReader(f,delimiter='\t'):
   q=str(r['query-id'])
   if q in qids and int(r['score'])>0: out.setdefault(q,{})[str(r['corpus-id'])]=int(r['score'])
 return out

def per_query(qids, rankings, rel):
 result={}
 for q,got in zip(qids,rankings):
  rr=rel.get(q,{})
  gains=np.asarray([2**rr.get(d,0)-1 for d in got],dtype=np.float64)
  dcg=float(np.sum(gains/np.log2(np.arange(2,len(got)+2))))
  ideal=sorted((2**x-1 for x in rr.values()),reverse=True)[:10]
  idcg=float(np.sum(np.asarray(ideal)/np.log2(np.arange(2,len(ideal)+2)))) if ideal else 0.
  hit=next((i+1 for i,d in enumerate(got) if d in rr),None)
  result[q]={'nDCG@10':dcg/idcg if idcg else 0.,'Recall@10':float(hit is not None),'MRR@10':1./hit if hit else 0.}
 return result

def main():
 split=json.loads((HERE/'runs/SPLIT_LOCK.json').read_text()); lock=json.loads((HERE/'DEV_LOCK.json').read_text())
 audit=json.loads((HERE/'runs/AUDIT_RESULTS.json').read_text()); top=json.loads((HERE/'runs/audit_top10.json').read_text()); saved={r['arm']:r for r in audit['rows']}
 assert not(set(split['dev_qids'])&set(split['audit_qids']))
 assert set(split['dev_qids'])|set(split['audit_qids'])==set(json.loads((HERE/'runs/ids.json').read_text())['query_ids'])
 assert hashlib.sha256('\n'.join(sorted(split['dev_qids'])).encode()).hexdigest()==split['dev_qids_sha256']
 assert hashlib.sha256('\n'.join(sorted(split['audit_qids'])).encode()).hexdigest()==split['audit_qids_sha256']
 assert split['role_threshold_wordpieces']==lock['role_threshold_wordpieces']
 # Tie-breaking must recover the development-locked angles.
 for role,scores in lock['dev_ndcg10_grid'].items():
  choices=sorted(range(len(GRID)),key=lambda i:(-scores[i],abs(float(GRID[i])),float(GRID[i])))
  assert float(GRID[choices[0]])==lock['angles_rad'][role]
 qids=top['qids']; assert len(qids)==705 and len(set(qids))==len(qids)
 assert set(qids)<=set(split['audit_qids']) and not(set(qids)&set(split['dev_qids']))
 rel=rels_for(set(qids)); assert len(rel)==len(qids)
 stored=top['top10']; assert set(stored)=={'native','mirror','linear_control'}
 assert stored['mirror']==stored['linear_control']
 doc_ids=json.loads((HERE/'runs/ids.json').read_text())['doc_ids']
 raw={arm:[[doc_ids[int(j)] for j in row] for row in stored[arm]] for arm in stored}
 # Replay every saved ANN ranking from the serialized ScaNN index and frozen vectors.
 all_ids=json.loads((HERE/'runs/ids.json').read_text())['query_ids']; qix={q:i for i,q in enumerate(all_ids)}
 qi=np.asarray([qix[q] for q in qids],dtype=np.int64); Q=np.load(HERE/'runs/query_vectors.npy',allow_pickle=False)[qi]
 lens=np.load(HERE/'runs/query_lengths.npy',allow_pickle=False)[qi]; roles=np.where(lens<=lock['role_threshold_wordpieces'],'short','long')
 searcher=scann.scann_ops_pybind.load_searcher(str(HERE/'payload/scann_index'))
 mapped={'native':Q,'mirror':np.vstack([givens(Q[i:i+1],lock['angles_rad'][roles[i]]) for i in range(len(Q))]),'linear_control':np.vstack([linear(Q[i:i+1],lock['angles_rad'][roles[i]]) for i in range(len(Q))])}
 assert np.array_equal(mapped['mirror'],mapped['linear_control'])
 ranking_replay={}
 for arm in ('native','mirror','linear_control'):
  ranked=searcher.search_batched(mapped[arm],final_num_neighbors=10,pre_reorder_num_neighbors=100,leaves_to_search=16)[0]
  assert np.array_equal(ranked,np.asarray(stored[arm],dtype=np.int32)),arm
  ranking_replay[arm]=True
 # Recompute exact dense cosine top-10 and the ANN-overlap metric from retained vectors.
 corpus=np.load(HERE/'runs/corpus_vectors.npy',allow_pickle=False)
 dense=mapped['native']@corpus.T; exact_ix=np.argpartition(-dense,9,axis=1)[:,:10]
 er=np.arange(len(qids))[:,None]; exact_score=dense[er,exact_ix]
 exact_ix=exact_ix[er,np.argsort(-exact_score,axis=1,kind='stable')]
 overlap={}
 for arm in ('native','mirror','linear_control'):
  got=np.asarray(stored[arm],dtype=np.int64)
  val=float(np.mean([len(set(map(int,got[i]))&set(map(int,exact_ix[i])))/10 for i in range(len(qids))]))
  target=saved[arm]['ann_overlap_recall10_vs_exact']
  assert val==target,(arm,val,target)
  overlap[arm]=val
 replay={arm:per_query(qids,raw[arm],rel) for arm in raw}
 maxdiff=0.; rows=[]
 for arm in ('native','mirror','linear_control'):
  for metric in ('nDCG@10','Recall@10','MRR@10'):
   val=float(np.mean([replay[arm][q][metric] for q in qids])); delta=abs(val-saved[arm][metric]); maxdiff=max(maxdiff,delta); assert delta==0.,(arm,metric,val,saved[arm][metric])
  rows.append({'arm':arm,'metrics_replayed':saved[arm]})
 # Paired bootstrap uncertainty is descriptive only; no gate or tuning uses it.
 rng=np.random.default_rng(1040); n=len(qids); boot={}
 for metric in ('nDCG@10','Recall@10','MRR@10'):
  a=np.asarray([replay['mirror'][q][metric] for q in qids]); b=np.asarray([replay['native'][q][metric] for q in qids]); dif=[]
  for _ in range(10000):
   ix=rng.integers(0,n,n); dif.append(float(np.mean(a[ix]-b[ix])))
  boot[metric]={'mean_delta':float(a.mean()-b.mean()),'paired_bootstrap_95pct_percentile_ci':[float(np.quantile(dif,.025)),float(np.quantile(dif,.975))],'replicates':10000,'seed':1040}
 # Recompute every paid bundle size and payload file digest.
 model_names=['onnx/model_quantized.onnx','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']
 model_files=[{'path':f,'bytes':(MODEL/f).stat().st_size,'sha256':sha(MODEL/f)} for f in model_names if (MODEL/f).exists()]
 model_bytes=sum(f['bytes'] for f in model_files)
 index=[]
 for p in sorted((HERE/'payload/scann_index').rglob('*')):
  if p.is_file(): index.append({'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)})
 assert index==audit['payload']['index_files']
 index_bytes=sum(f['bytes'] for f in index); idbytes=(HERE/'payload/doc_ids.json').stat().st_size
 viewbytes=(HERE/'payload/view_config.json').stat().st_size; linearbytes=(HERE/'payload/linear_control.json').stat().st_size
 assert (model_bytes,index_bytes,idbytes,viewbytes,linearbytes)==(audit['payload']['model_bytes'],audit['payload']['index_bytes'],audit['payload']['doc_id_map_bytes'],audit['payload']['mirror_code_and_role_metadata_bytes'],audit['payload']['linear_control_and_role_metadata_bytes'])
 assert model_bytes+index_bytes+idbytes+viewbytes==audit['payload']['mirror_full_payload_bytes']
 assert model_bytes+index_bytes+idbytes+linearbytes==audit['payload']['linear_control_full_payload_bytes']
 assert model_bytes+index_bytes+idbytes==audit['payload']['native_full_payload_bytes']
 data={'archive_sha256':sha(Path('/tmp/ma1040_arguana.zip')),'corpus_sha256':sha(DATA/'corpus.jsonl'),'queries_sha256':sha(DATA/'queries.jsonl'),'qrels_sha256':sha(DATA/'qrels/test.tsv')}
 assert data['qrels_sha256']==audit['qrels_file_sha256']
 bootout={'fresh_query_count':n,'max_metric_difference':maxdiff,'metrics_exact':True,'split_disjoint':True,'split_union_exact':True,'dev_angle_selection_tie_break_replayed':True,'mirror_linear_top10_exact':True,'scann_rankings_replayed_exact':ranking_replay,'exact_cosine_overlap_replayed':overlap,'payload_bytes_exact':True,'payload_hashes_exact':True,'paired_bootstrap':boot,'dataset_hashes':data,'model_files':model_files,'scann_index_files':index}
 (HERE/'runs/VERIFICATION_REPLAY.json').write_text(json.dumps(bootout,indent=2)+'\n')
 print(json.dumps({'passed':True,'fresh_queries':n,'max_metric_difference':maxdiff,'payload_bytes_exact':True,'paired_bootstrap':boot},indent=2))
if __name__=='__main__': main()
