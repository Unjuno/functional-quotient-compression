#!/usr/bin/env python3
"""Read-only selfcheck: 2 MA mechanism pilots + public tabular follow-up, no network."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RESULT=ROOT/'results'
SOURCES=('ma1171_joint_codec.py','test_ma1171.py','ma1183_tabm_style.py','test_ma1183.py','ma1183_real_tabular.py','test_ma1183_real.py')
CORE=('MA1171_RESULTS_CORE.csv','MA1183_SYNTHETIC_RESULTS_CORE.csv','MA1183_REAL_RESULTS_CORE.csv')
RAW={
 'MA1171_DEV_RAW.csv':(108,'seed'),
 'MA1171_FRESH_RAW.csv':(180,'seed'),
 'MA1171_FRESH_COMPARISON.csv':(45,'seed'),
 'MA1183_SYNTH_DEV_RAW.csv':(45,'seed'),
 'MA1183_SYNTH_FRESH_RAW.csv':(75,'seed'),
 'MA1183_REAL_RAW.csv':(50,'seed'),
}

def check():
 problems=[]
 def fail(msg):problems.append(msg)
 for name in SOURCES:
  if not (ROOT/'source'/name).is_file():fail('missing source '+name)
 for name in ('MA1171_FROZEN_PROTOCOL.json','MA1183_FROZEN_PROTOCOL.json','MA1183_REAL_TABULAR_FROZEN_PROTOCOL.json','AMENDMENT_BEFORE_FRESH.json','REPORT.md','REPRODUCE.md'):
  if not (ROOT/name).is_file():fail('missing source protocol '+name)
 for name in CORE:
  if not (RESULT/name).is_file():fail('missing core '+name)
 for name,(expected,seed_key) in RAW.items():
  p=RESULT/name
  if not p.is_file():fail('missing raw '+name);continue
  with p.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
  if len(rows)!=expected:fail(f'{name} row count {len(rows)} != {expected}')
  got={int(r[seed_key]) for r in rows}
  expected_ids={11,12,13} if 'DEV' in name else ({301,302,303,304,305} if 'REAL' in name else {101,102,103,104,105})
  if got!=expected_ids:fail(name+' wrong seeds '+str(got))
 core=json.loads((RESULT/'VERIFICATION.json').read_text(encoding='utf-8'))
 filemap={
 'ma1171_joint_codec.py':ROOT/'source/ma1171_joint_codec.py',
 'test_ma1171.py':ROOT/'source/test_ma1171.py',
 'ma1183_tabm_style.py':ROOT/'source/ma1183_tabm_style.py',
 'test_ma1183.py':ROOT/'source/test_ma1183.py',
 'ma1183_real_tabular.py':ROOT/'source/ma1183_real_tabular.py',
 'test_ma1183_real.py':ROOT/'source/test_ma1183_real.py',
 'ma1171_results/dev_raw.csv':RESULT/'MA1171_DEV_RAW.csv',
 'ma1171_results/fresh_raw.csv':RESULT/'MA1171_FRESH_RAW.csv',
 'ma1171_results/fresh_comparison.csv':RESULT/'MA1171_FRESH_COMPARISON.csv',
 'ma1183_results/dev_raw.csv':RESULT/'MA1183_SYNTH_DEV_RAW.csv',
 'ma1183_results/fresh_raw.csv':RESULT/'MA1183_SYNTH_FRESH_RAW.csv',
 'ma1183_real_results/real_raw.csv':RESULT/'MA1183_REAL_RAW.csv',
 'REPLAY_VERIFICATION.json':RESULT/'REPLAY_VERIFICATION.json',
 'results/MA1171_RESULTS_CORE.csv':RESULT/'MA1171_RESULTS_CORE.csv',
 'results/MA1183_SYNTHETIC_RESULTS_CORE.csv':RESULT/'MA1183_SYNTHETIC_RESULTS_CORE.csv',
 'results/MA1183_REAL_RESULTS_CORE.csv':RESULT/'MA1183_REAL_RESULTS_CORE.csv',
 }
 for p,sha in core['hash_sha256'].items():
  if p not in filemap:fail('unknown source hash '+p);continue
  v=filemap[p]
  if not v.is_file():fail('missing '+str(v));continue
  h=hashlib.sha256(v.read_bytes()).hexdigest()
  if sha!=h:fail('sha mismatch '+p)
 replay=json.loads((RESULT/'REPLAY_VERIFICATION.json').read_text(encoding='utf-8'))
 if not replay['MA-1171']['deterministic_replay_pass'] or not replay['MA-1183']['deterministic_replay_pass']:
  fail('replay manifest FAIL')
 if core['unit_tests']['all_passed'] is not True or core['corrected_worker_branch_untouched'] is not True:
  fail('verification status problem')
 print(json.dumps({'MA1171_rows':180,'MA1183_synth_rows':75,'MA1183_real_rows':50,'source_checksums':len(core['hash_sha256']),'errors':problems},ensure_ascii=False))
 return not problems

if __name__=='__main__':sys.exit(0 if check() else 1)
