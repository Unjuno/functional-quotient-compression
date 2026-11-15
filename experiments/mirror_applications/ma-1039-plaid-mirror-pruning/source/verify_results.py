#!/usr/bin/env python3
"""Independent replay of MA-1039 audit metrics and actual serialized bytes."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

EXP=Path(__file__).resolve().parents[1]
DATA=Path('/tmp/ma1039/trec-covid/trec-covid')
MODEL=Path('/tmp/ma1038/colbertv2.0')
INDEX=Path('/tmp/ma1039/run/ma1039/indexes/trec-covid-positive-pool-1bit-512c')
PAYLOAD=Path('/tmp/ma1039/payload')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def metric(qid,ranking,qrels,k=10):
    rel=qrels[qid]
    got=[rel.get(d,0) for d in ranking[:k]]
    dcg=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(got))
    ideal=sorted(rel.values(),reverse=True)[:k]
    idcg=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
    ndcg=dcg/idcg if idcg else 0.0
    rr=next((1.0/(i+1) for i,r in enumerate(got) if r>0),0.0)
    return ndcg,rr

def main():
    summary=json.loads((EXP/'runs/summary.json').read_text())
    freeze=json.loads((EXP/'runs/DEV_FREEZE.json').read_text())
    replay=json.loads((EXP/'runs/audit_rankings.json').read_text())
    qrels={}
    with (DATA/'qrels/test.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            qrels.setdefault(str(r['query-id']),{})[str(r['corpus-id'])]=int(r['score'])
    qrels={q:r for q,r in qrels.items() if any(v>0 for v in r.values())}
    roles={q:(0 if summary['audit_role_counts']['short'] and
                    q in set(x['qid'] for x in replay['plaid_native'] if x['qid'] in freeze['audit_queries']) and
                    # Raw role lengths are frozen in the runner; audit role assignment is rederived below.
                    False else 0) for q in []}
    # Reconstruct audit roles using the pinned ColBERT tokenizer and frozen development threshold.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    qmap={}
    with (DATA/'queries.jsonl').open() as f:
        for line in f:
            x=json.loads(line); qmap[str(x['_id'])]=str(x.get('text') or '')
    roles={q:(0 if len(tok.tokenize(qmap[q]))<=freeze['role_threshold_raw_tokens'] else 1) for q in freeze['audit_queries']}
    replay_metrics={}; replay_union={}; max_metric_delta=0.0
    for arm,rows in replay.items():
        nvals=[]; rvals=[]; recalls=[]; union=[]
        for row in rows:
            q=row['qid']; rank=row['ranking_top1000']
            n,r=metric(q,rank,qrels)
            nvals.append(n); rvals.append(r)
            positives={d for d,v in qrels[q].items() if v>0}
            recalls.append(len(positives.intersection(rank[:1000]))/len(positives) if positives else 0.0)
            if arm!='full_index_scan': union.append(len(positives.intersection(row['candidate_ids']))/len(positives) if positives else 0.0)
        replay_metrics[arm]={'nDCG@10':float(np.mean(nvals)),'MRR@10':float(np.mean(rvals)),'Recall@1000':float(np.mean(recalls)),'n':len(rows)}
        for m,v in replay_metrics[arm].items():
            if m!='n': max_metric_delta=max(max_metric_delta,abs(v-summary['metrics'][arm]['all'][m]))
        if union: replay_union[arm]=float(np.mean(union))
    mirror={r['qid']:r for r in replay['mirror']}
    native={r['qid']:r for r in replay['native_query_control']}
    exact_alias=all(mirror[q]['candidate_ids']==native[q]['candidate_ids'] and mirror[q]['ranking_top1000']==native[q]['ranking_top1000'] for q in mirror)
    archive_sha=sha(Path('/tmp/ma1039/trec-covid.zip'))
    model_files=sorted(p for p in MODEL.rglob('*') if p.is_file())
    index_files=sorted(p for p in INDEX.rglob('*') if p.is_file())
    index_bytes=sum(p.stat().st_size for p in index_files)
    model_bytes=sum(p.stat().st_size for p in model_files)
    pid_bytes=(PAYLOAD/'pid_to_docid.json').stat().st_size
    angle_bytes=(PAYLOAD/'role_angles.f32').stat().st_size+(PAYLOAD/'role_view.json').stat().st_size
    complete=index_bytes+model_bytes+pid_bytes
    payload_ok=(complete==summary['payload']['native']['complete_bytes'] and complete+angle_bytes==summary['payload']['mirror']['complete_bytes'] and angle_bytes==summary['payload']['marginal_view_bytes'])
    result={
        'experiment_id':'MA-1039','fresh_queries':len(replay['plaid_native']),
        'dataset_archive_sha256':archive_sha,'dataset_hash_matches':archive_sha==summary['dataset_archive_sha256'],
        'metric_replay_max_abs_difference':max_metric_delta,'metric_replay_exact':max_metric_delta==0.0,
        'replayed_metrics':replay_metrics,'replayed_prepruning_ivf_union_recall':replay_union,
        'prepruning_union_matches':all(abs(replay_union[k]-summary['prepruning_ivf_union_recall_mean'][k])<1e-12 for k in replay_union),
        'mirror_equals_native_query_control_all_audit':exact_alias,
        'centroid_score_max_abs_delta':summary['equivalence']['max_abs_centroid_score_delta'],
        'payload_bytes_replayed':{'index':index_bytes,'model_and_tokenizer':model_bytes,'pid_map':pid_bytes,'view':angle_bytes,'native_complete':complete,'mirror_complete':complete+angle_bytes},
        'payload_replay_exact':payload_ok,'index_file_sha256':{p.relative_to(INDEX).as_posix():sha(p) for p in index_files},
        'checkpoint_file_sha256':{p.relative_to(MODEL).as_posix():sha(p) for p in model_files},
        'roles':{'short':sum(roles.values()==0 for _ in []),'long':0},
        'notes':['Independent CPU replay from stored per-query rankings, official qrels and actual filesystem payloads.']}
    result['roles']={'short':sum(1 for q in roles if roles[q]==0),'long':sum(1 for q in roles if roles[q]==1)}
    (EXP/'runs/verification_replay.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    if not result['dataset_hash_matches'] or not result['metric_replay_exact'] or not result['prepruning_union_matches'] or not result['mirror_equals_native_query_control_all_audit'] or not result['payload_replay_exact']:
        raise SystemExit('MA-1039 verification failed')

if __name__=='__main__': main()
