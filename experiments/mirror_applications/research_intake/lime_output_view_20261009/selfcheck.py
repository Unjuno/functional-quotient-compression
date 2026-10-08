#!/usr/bin/env python3
"""Read-only selfcheck of isolated LME01 scientific intake on FULL git checkout."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv,json,re,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
APPS=ROOT/'experiments'/'mirror_applications'
ERR=[]
def ensure(ok,msg):
    if not ok:ERR.append(msg)
def records(p):
    with p.open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))
rows=records(APPS/'IDEA_REGISTRY.csv')
ids=[r.get('id') for r in rows]
ensure(ids==[f'MA-{i:03d}' for i in range(1,1194)],'MA gap/duplicate/schema error')
pri=Counter(r.get('priority') for r in rows)
status=Counter(r.get('status') for r in rows)
ensure(pri==Counter({'P0':651,'P1':439,'P2':103}),f'bad priorities {dict(pri)}')
ensure(status==Counter({'UNTESTED':1146,'PROMISING':29,'FAIL':18}),f'bad statuses {dict(status)}')
text=(ROOT/'docs'/'phase2'/'MIRROR_APPLICATION_PRIOR_ART.md').read_text(encoding='utf-8')
papers=[int(x) for x in re.findall(r'^## PA(\d+)\b',text,re.M)]
ensure(papers==list(range(1,471)),'PA missing/duplicate IDs')
for r in rows:
    for n in re.findall(r'PA(\d+)',r.get('prior_art_refs','')):
        ensure(int(n) in papers,f'{r["id"]} orphan PA{n}')
claims=records(APPS/'CLAIM_LEDGER.csv')
byid={r['id']:r for r in rows}
ensure(len(claims)==47,'claims should be 47')
for c in claims:
    ensure(byid.get(c.get('id'),{}).get('status')==c.get('status'),f'claim mismatch {c.get("id")}')
board=(APPS/'STATUS_BOARD.md').read_text(encoding='utf-8')
ensure('Registered candidates: **1193**' in board,'status board count not updated')
ensure('1146 UNTESTED' in board,'status board status not updated')
for num in range(1189,1194):
    r=byid[f'MA-{num}']
    ensure(r['status']=='UNTESTED','new candidate has results status')
    d=HERE/'plans'/f'MA-{num}'
    for n in ('README.md','PROTOCOL.json','STATUS.md'):
        ensure((d/n).is_file(),f'plan missing MA-{num}/{n}')
    if not (d/'PROTOCOL.json').exists():continue
    p=json.loads((d/'PROTOCOL.json').read_text(encoding='utf-8'))
    ensure(p['experiment_id']==f'MA-{num}' and p['worker_claim']==False,'bad plan config')
    ensure(p['splits']['development']['seeds']==[11,12,13],'bad dev seeds')
    ensure(p['splits']['fresh']['seeds']==[101,102,103,104,105],'bad fresh seeds')
    ensure(all(x in p['gates'] for x in ('PASS','FAIL','UNCERTAIN')),'missing decisions')
    prose=(d/'README.md').read_text(encoding='utf-8')
    ensure(all(x in prose for x in ('## H','## T','## D','## C','## U','Mirror insertion')),'missing standalone prose')
    ensure(not (d/'RESULTS_CORE.csv').exists(), 'plan-only file is completed')
for num in (1175,1178,1167,1105,1185,1176,1173):
    ensure((HERE/'existing'/f'MA-{num}.md').is_file(),f'missing existing control {num}')
pilot=HERE/'pilots'/'lme01'
verify=json.loads((pilot/'results'/'VERIFICATION.json').read_text(encoding='utf-8'))
replay=json.loads((pilot/'results'/'REPLAY_VERIFICATION.json').read_text(encoding='utf-8'))
ensure(verify['dev_rows']==21 and verify['fresh_rows']==35,'data row missing')
ensure(verify['mirror_rot_001_gain_worlds']==1,'scientific gate changed')
ensure(replay['match'] and replay['cells_compared']==154,'replay inconsistent')
print(f'MA={len(rows)}, PA={len(papers)}, claims={len(claims)}, P0={pri["P0"]}, P1={pri["P1"]}, P2={pri["P2"]}, UNTESTED={status["UNTESTED"]}, errors={len(ERR)}')
for x in ERR: print('ERROR',x,file=sys.stderr)
sys.exit(1 if ERR else 0)
