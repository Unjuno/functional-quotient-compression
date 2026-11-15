#!/usr/bin/env python3
"""MA-1039: official ColBERT CPU PLAID traversal with a centroid View adapter."""
from __future__ import annotations

import csv, hashlib, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
EXTERNAL = Path("/tmp/ma1038/ColBERT")
DEPS = Path("/tmp/ma1039deps")
MODEL = Path("/tmp/ma1038/colbertv2.0")
DATA = Path("/tmp/ma1039/trec-covid")
OUT = Path("/tmp/ma1039/run")
CAP = 512
SEED = 1039
ANGLE_GRID = np.asarray([-0.20,-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20],dtype=np.float32)
if str(DEPS) not in sys.path: sys.path.insert(0,str(DEPS))
if str(EXTERNAL) not in sys.path: sys.path.insert(0,str(EXTERNAL))
os.environ["PATH"] = str(DEPS/"bin") + os.pathsep + os.environ.get("PATH","")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
torch.set_num_threads(4); torch.set_num_interop_threads(1)

from plaid_views import install_view_adapter, configure_state


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_jsonl(path):
    with open(path) as f:
        for line in f: yield json.loads(line)

def load_data():
    root=DATA/"trec-covid"
    corpus=list(read_jsonl(root/"corpus.jsonl"))
    queries=list(read_jsonl(root/"queries.jsonl"))
    qmap={str(x["_id"]):str(x.get("text") or "") for x in queries}
    docs={str(x["_id"]):((str(x.get("title") or "")+" "+str(x.get("text") or "")).strip()) for x in corpus}
    qrels={}
    with (root/"qrels"/"test.tsv").open() as f:
        for row in csv.DictReader(f,delimiter="\t"):
            qrels.setdefault(str(row["query-id"]),{})[str(row["corpus-id"])]=int(row["score"])
    qrels={q:r for q,r in qrels.items() if q in qmap and any(v>0 for v in r.values())}
    dev=sorted(q for q in qrels if int(q)%5 not in (0,1))
    audit=sorted(q for q in qrels if int(q)%5 in (0,1))
    positives={d for rel in qrels.values() for d,v in rel.items() if v>0 and d in docs}
    bg=[d for d in docs if d not in positives]
    bg.sort(key=lambda d:hashlib.sha256(f"MA1039|{SEED}|{d}".encode()).hexdigest())
    bg=bg[:512]
    docids=sorted(positives.union(bg))
    texts=[docs[d] for d in docids]
    return docids,texts,qmap,qrels,dev,audit,corpus,queries

def patch_small_corpus_partition_cap():
    """Bound only upstream's automatic partition count on this small screen."""
    from colbert.indexing.collection_indexer import CollectionIndexer
    original=CollectionIndexer.setup
    original_sample=CollectionIndexer._sample_pids
    if getattr(CollectionIndexer,"_ma1039_cap_installed",False): return
    def setup_with_cap(self):
        original(self)
        actual=min(CAP,int(self.num_partitions))
        if actual!=self.num_partitions:
            self.num_partitions=actual
            self._save_plan()
    def bounded_deterministic_sample(self):
        # 4,096 passages provide hundreds of thousands of token vectors for
        # the 512-centroid CPU screen while avoiding ColBERT's all-doc sample
        # on this 18k-document bounded collection.
        n=min(4096,len(self.collection))
        return random.Random(SEED).sample(range(len(self.collection)),n)
    CollectionIndexer.setup=setup_with_cap
    CollectionIndexer._sample_pids=bounded_deterministic_sample
    CollectionIndexer._ma1039_cap_installed=True

def make_or_load_index(docids,texts):
    import faiss
    from colbert import Indexer
    from colbert.infra import ColBERTConfig,Run,RunConfig
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); faiss.omp_set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    name="trec-covid-positive-pool-1bit-512c"
    expected=OUT/"ma1039"/"indexes"/name
    if (expected/"metadata.json").exists():
        return name,str(expected),0.0
    patch_small_corpus_partition_cap()
    cfg=ColBERTConfig(root=str(OUT),experiment="ma1039",nbits=1,dim=128,doc_maxlen=180,query_maxlen=32,mask_punctuation=True,kmeans_niters=20,index_bsize=64,nranks=1,avoid_fork_if_possible=True)
    started=time.perf_counter()
    with Run().context(RunConfig(nranks=1,experiment="ma1039",root=str(OUT),avoid_fork_if_possible=True)):
        ix=Indexer(checkpoint=str(MODEL),config=cfg,verbose=1)
        index_path=ix.index(name=name,collection=texts,overwrite="force_silent_overwrite")
    index_path=Path(index_path)
    if not (index_path/"metadata.json").exists():
        raise RuntimeError(f"Official Indexer returned an index without metadata: {index_path}")
    return name,str(index_path),time.perf_counter()-started

def metric(qid,ranking,qrels,k=10):
    rel=qrels[qid]
    got=[rel.get(d,0) for d in ranking[:k]]
    dcg=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(got))
    ideal=sorted(rel.values(),reverse=True)[:k]
    idcg=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
    ndcg=dcg/idcg if idcg else 0.0
    rr=next((1.0/(i+1) for i,r in enumerate(got) if r>0),0.0)
    return ndcg,rr

def run_one(searcher,state,Q,docids,qid,mode,angle=0.0,k=100,full=False):
    configure_state(state,mode,angle)
    n=len(docids)
    pids=torch.arange(n,dtype=torch.int32) if full else None
    t=time.perf_counter()
    ranked, scores=searcher.ranker.rank(searcher.config,Q,pids=pids)
    elapsed=time.perf_counter()-t
    full_ranking=[docids[int(i)] for i in ranked]
    ranking=full_ranking[:k]
    candidates=state["candidate_pids"]
    candidate_ids=[] if candidates is None else [docids[int(i)] for i in candidates.tolist()]
    if full:
        candidate_ids=list(docids)
    cells=state["cells"]
    return {"ranking":ranking,"ranking_full":full_ranking,"scores":scores[:k],"candidate_ids":candidate_ids,"candidate_count":len(candidate_ids),"cells":[] if cells is None else cells.tolist(),"centroid_scores":state["centroid_scores"],"route_seconds":state["route_seconds"],"wall_seconds":elapsed}

def summarize(rows,qrels,roles):
    qids=[r["qid"] for r in rows]
    groups={"all":qids}
    for role in sorted({roles[q] for q in qids}): groups[f"role_{role}"]=[q for q in qids if roles[q]==role]
    result={}
    for group,qs in groups.items():
        vals=[metric(q,next(x["ranking"] for x in rows if x["qid"]==q),qrels) for q in qs]
        recs=[]
        for q in qs:
            row=next(x for x in rows if x["qid"]==q)
            relevant={d for d,v in qrels[q].items() if v>0}
            top1000=row["ranking_full"][:1000]
            recs.append(len(relevant.intersection(top1000))/len(relevant) if relevant else 0.0)
        result[group]={"n":len(qs),"nDCG@10":float(np.mean([x[0] for x in vals])),"MRR@10":float(np.mean([x[1] for x in vals])),"Recall@1000":float(np.mean(recs))}
    return result

def payload_sizes(index_path,docids,model_files,view_angles=None,threshold=None):
    index_path=Path(index_path)
    index_files={p.relative_to(index_path).as_posix():p.stat().st_size for p in sorted(index_path.rglob('*')) if p.is_file()}
    payload=Path("/tmp/ma1039/payload"); payload.mkdir(parents=True,exist_ok=True)
    for p in model_files:
        dest=payload/p.name
        if not dest.exists():
            import shutil; shutil.copy2(p,dest)
    map_path=payload/"pid_to_docid.json"
    map_path.write_text(json.dumps(docids,separators=(",",":")))
    view_bytes=0
    if view_angles is not None:
        raw=np.asarray(view_angles,dtype='<f4').tobytes()
        p=payload/"role_angles.f32"; p.write_bytes(raw)
        meta={"role_threshold_raw_tokens":int(threshold),"short_inclusive":True,"view":"shared 64-plane Givens on centroid scoring bank"}
        m=payload/"role_view.json"; m.write_text(json.dumps(meta,separators=(",",":")))
        view_bytes=p.stat().st_size+m.stat().st_size
    model_sizes={p.name:p.stat().st_size for p in model_files}
    return {"index_bytes":sum(index_files.values()),"index_files":index_files,"pid_map_bytes":map_path.stat().st_size,"model_bytes":sum(model_sizes.values()),"model_files":model_sizes,"view_bytes":view_bytes,"complete_bytes":sum(index_files.values())+map_path.stat().st_size+sum(model_sizes.values())+view_bytes}

def main():
    if not MODEL.exists() or not DATA.exists(): raise SystemExit("Pinned ColBERTv2 checkpoint or TREC-COVID dataset is missing")
    docids,texts,qmap,qrels,dev_qids,aud_qids,corpus,queries=load_data()
    print(f"candidate docs={len(docids)}; dev={len(dev_qids)}; audit={len(aud_qids)}; all-query-positive-docs={sum(1 for d in docids if any(d in x and x[d]>0 for x in qrels.values()))}",flush=True)
    name,index_path,index_wall=make_or_load_index(docids,texts)
    from colbert import Searcher
    from colbert.infra import ColBERTConfig,Run,RunConfig
    search_cfg=ColBERTConfig(root=str(OUT),experiment="ma1039",ncells=2,centroid_score_threshold=0.45,ndocs=4096,query_maxlen=32)
    with Run().context(RunConfig(nranks=1,experiment="ma1039_search",root=str(OUT))):
        searcher=Searcher(index=name,checkpoint=str(MODEL),collection=texts,config=search_cfg,index_root=str(Path(index_path).parent),verbose=1)
    state=install_view_adapter(searcher.ranker)
    tokenizer=searcher.checkpoint.query_tokenizer
    query_token_count={q:len(tokenizer.tokenize([qmap[q]])[0]) for q in qrels}
    threshold=int(np.median([query_token_count[q] for q in dev_qids]))
    roles={q:(0 if query_token_count[q]<=threshold else 1) for q in qrels}
    Qcache={}
    for q in dev_qids+aud_qids: Qcache[q]=searcher.encode(q)
    # Freeze one role coordinate per role on development MRR@10 only.
    dev_rows={}
    for q in dev_qids:
        r=run_one(searcher,state,Qcache[q],docids,q,"native",k=100)
        dev_rows[q]=r
    angle_scores={0:[],1:[]}
    for a in ANGLE_GRID:
        for q in dev_qids:
            rr=run_one(searcher,state,Qcache[q],docids,q,"mirror_centroid_bank",float(a),k=100)
            dev_rows[(q,float(a))]=rr
        for role in [0,1]:
            qs=[q for q in dev_qids if roles[q]==role]
            vals=[metric(q,dev_rows[(q,float(a))]["ranking"],qrels)[1] for q in qs]
            angle_scores[role].append((float(np.mean(vals)),float(a)))
    chosen=[]
    for role in [0,1]:
        best=max(x[0] for x in angle_scores[role]); tied=[x[1] for x in angle_scores[role] if abs(x[0]-best)<1e-12]
        chosen.append(sorted(tied,key=lambda a:(abs(a),a))[0])
    freeze={"role_threshold_raw_tokens":threshold,"angles":chosen,"angle_grid":ANGLE_GRID.tolist(),"dev_angle_mrr":{str(k):v for k,v in angle_scores.items()},"dev_queries":dev_qids,"audit_queries":aud_qids,"selection":"dev qrels only; frozen before audit scoring"}
    (EXP/"runs").mkdir(exist_ok=True)
    (EXP/"runs"/"DEV_FREEZE.json").write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n")
    # Fresh query audit: native PLAID, centroid View, native query push-through, and full compressed scan.
    arms={"plaid_native":[],"mirror":[],"native_query_control":[],"full_index_scan":[]}
    candidate_recall={"plaid_native":[],"mirror":[],"native_query_control":[]}
    equivalence=[]; audit_times=[]
    for q in aud_qids:
        base=run_one(searcher,state,Qcache[q],docids,q,"native",k=100)
        mir=run_one(searcher,state,Qcache[q],docids,q,"mirror_centroid_bank",chosen[roles[q]],k=100)
        nat=run_one(searcher,state,Qcache[q],docids,q,"native_query_routing",chosen[roles[q]],k=100)
        full=run_one(searcher,state,Qcache[q],docids,q,"native",k=100,full=True)
        for name,row in [("plaid_native",base),("mirror",mir),("native_query_control",nat),("full_index_scan",full)]:
            row["qid"]=q; arms[name].append(row)
            if name!="full_index_scan":
                rel={d for d,v in qrels[q].items() if v>0}
                candidate_recall[name].append(len(rel.intersection(row["candidate_ids"]))/len(rel) if rel else 0.0)
        cell_scores=float((mir["centroid_scores"]-nat["centroid_scores"]).abs().max().item()) if mir["centroid_scores"] is not None else 0.0
        equivalence.append({"qid":q,"same_cells":mir["cells"]==nat["cells"],"same_candidate_ids":mir["candidate_ids"]==nat["candidate_ids"],"max_abs_centroid_score_delta":cell_scores,"same_top100":mir["ranking"]==nat["ranking"]})
    audit_roles={q:roles[q] for q in aud_qids}
    metrics={name:summarize(rows,qrels,audit_roles) for name,rows in arms.items()}
    top10_changed=sum(arms["plaid_native"][i]["ranking"][:10]!=arms["mirror"][i]["ranking"][:10] for i in range(len(aud_qids)))
    model_files=[MODEL/n for n in ["pytorch_model.bin","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.txt"] if (MODEL/n).exists()]
    base_payload=payload_sizes(index_path,docids,model_files)
    view_payload=payload_sizes(index_path,docids,model_files,chosen,threshold)
    # View state's actual serialized files are counted once and shared with the native control.
    indexmeta=json.loads((Path(index_path)/"metadata.json").read_text())
    summary={"experiment":"MA-1039","dataset_archive_sha256":sha256(DATA/"trec-covid.zip"),"checkpoint_archive_sha256":"d07df4b2a9fe1848fbed333127a1a5b4f2d96106f420948f0db4450b018cb119","colbert_reference_commit":"cc4f3dc91c0b45d2d08c251d9d95178285c65f1c","candidate_documents":len(docids),"development_queries":dev_qids,"audit_queries":aud_qids,"role_threshold_raw_tokens":threshold,"audit_role_counts":{"short":sum(roles[q]==0 for q in aud_qids),"long":sum(roles[q]==1 for q in aud_qids)},"angles":chosen,"angle_grid":ANGLE_GRID.tolist(),"development_angle_mrr":angle_scores,"metrics":metrics,"prepruning_ivf_union_recall_mean":{k:float(np.mean(v)) for k,v in candidate_recall.items()},"candidate_counts_mean":{k:float(np.mean([x["candidate_count"] for x in arms[k]])) for k in ["plaid_native","mirror","native_query_control"]},"ranking_divergence":{"top10_changed_queries":int(top10_changed),"query_count":len(aud_qids)},"equivalence":{"all_same_cells":all(x["same_cells"] for x in equivalence),"all_same_candidate_ids":all(x["same_candidate_ids"] for x in equivalence),"all_same_top100":all(x["same_top100"] for x in equivalence),"max_abs_centroid_score_delta":max(x["max_abs_centroid_score_delta"] for x in equivalence),"per_query":equivalence},"payload":{"native":base_payload,"mirror":view_payload,"native_query_control":view_payload,"marginal_view_bytes":view_payload["complete_bytes"]-base_payload["complete_bytes"]},"latency_status":"NOT ESTABLISHED: one pass per arm/query, no repeated isolated timing calibration", "index":{"path":str(index_path),"num_partitions":indexmeta["num_partitions"],"num_embeddings":indexmeta["num_embeddings"],"doclens_file_count":len(list(Path(index_path).glob("doclens.*.json"))),"build_wall_seconds":index_wall},"optimizer_updates":0,"development_angle_query_evaluations":len(ANGLE_GRID)*len(dev_qids),"audit_query_wall_seconds":{k:[float(x["wall_seconds"]) for x in arms[k]] for k in ["plaid_native","mirror","native_query_control","full_index_scan"]},"audit_route_seconds":{k:[float(x["route_seconds"]) for x in arms[k]] for k in ["plaid_native","mirror","native_query_control"]}}
    (EXP/"runs"/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True),flush=True)

if __name__=="__main__": main()
