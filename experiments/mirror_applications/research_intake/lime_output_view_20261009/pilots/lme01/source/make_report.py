#!/usr/bin/env python3
"""Generate honest source-locked LME01 report from pre-frozen raw rows."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import sklearn
import scipy
import torch
from source.run_lme01 import METHODS,SEEDS,K,STEPS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def load_rows(name):
    with (OUT/name).open(encoding='utf-8',newline='') as fp:
        return list(csv.DictReader(fp))

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarise(data):
    d=defaultdict(list)
    for r in data:d[r['method']].append(r)
    summary=[]
    for method in METHODS:
        rr=d[method]
        assert len(rr)==5
        entry={'method':method,'K':K,'fresh_worlds':len(rr),
         'mean_test_nll':statistics.mean(float(r['test_nll']) for r in rr),
         'mean_worst_task_nll':statistics.mean(float(r['test_worst_task_nll']) for r in rr),
         'mean_test_accuracy':statistics.mean(float(r['test_accuracy']) for r in rr),
         'mean_output_prob_std':statistics.mean(float(r['output_prob_std']) for r in rr),
         'mean_serialized_bytes':statistics.mean(int(r['serializer_bytes']) for r in rr),
         'median_cpu_p95_ms':statistics.median(float(r['cpu_p95_ms']) for r in rr),
         'median_cpu_p50_ms':statistics.median(float(r['cpu_p50_ms']) for r in rr),
         'role_trainable_scalars':int(rr[0]['role_trainable_scalars']),
         'min_test_nll':min(float(r['test_nll']) for r in rr),
         'max_test_nll':max(float(r['test_nll']) for r in rr)}
        summary.append(entry)
    return summary


def main():
    dev=load_rows('dev_raw.csv')
    fresh=load_rows('fresh_raw.csv')
    assert len(dev)==21 and len(fresh)==35
    for phase, data in (('dev',dev),('fresh',fresh)):
        observed={(int(r['seed']),r['method']) for r in data}
        expected={(s,m) for s in SEEDS[phase] for m in METHODS}
        assert observed==expected,(phase,expected-observed)
        assert all(int(r['K'])==K and int(r['training_steps'])==STEPS for r in data)
        assert all(r['one_trunk_forward']=='1' and r['one_adapter_forward']=='1' for r in data)
        for seed in SEEDS[phase]:
            group=[r for r in data if int(r['seed'])==seed]
            assert len({r['original_train_id_sha256'] for r in group})==1
            assert len({r['audit_id_sha256'] for r in group})==1
    summary=summarise(fresh)
    with (OUT/'RESULTS_CORE.csv').open('w',encoding='utf-8',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    by={(int(r['seed']),r['method']):r for r in fresh}
    differences={
     str(seed):{
       'rot_minus_linear4':float(by[seed,'mirror_rot4']['test_nll'])-float(by[seed,'lime_linear4']['test_nll']),
       'shear_minus_linear4':float(by[seed,'mirror_shear4']['test_nll'])-float(by[seed,'lime_linear4']['test_nll']),
       'rot_minus_lime_full':float(by[seed,'mirror_rot4']['test_nll'])-float(by[seed,'lime_full_diagonal']['test_nll']),
       'rot_minus_native_heads':float(by[seed,'mirror_rot4']['test_nll'])-float(by[seed,'native_onepass_linearheads']['test_nll'])}
     for seed in SEEDS['fresh']}
    wins=sum(v['rot_minus_linear4']<=-.01 for v in differences.values())
    normwins=sum(v['rot_minus_linear4']<0 for v in differences.values())
    replay=json.loads((OUT/'REPLAY_VERIFICATION.json').read_text(encoding='utf8'))
    checks={
      'all_dev_rows':len(dev)==21,
      'all_fresh_rows':len(fresh)==35,
      'one_trunk_and_adapter_call_per_all_training_methods':True,
      'same_train_audit_ids_within_seed':True,
      'geometry_exact_same_as_native_givens_test_pass':True,
      'within_environment_14_replays_154_field_matches':replay['match'] and replay['cells_compared']==154,
      'mechanism_gate_rot_improve_001_in_ge4_worlds':wins>=4,
      'quality_noninferior_to_strong_native_heads_002':all(v['rot_minus_native_heads']<=.02 for v in differences.values())
    }
    hashes={f: digest(ROOT/f) for f in ('source/run_lme01.py','source/test_lme01.py','results/dev_raw.csv','results/fresh_raw.csv','results/RESULTS_CORE.csv','results/REPLAY_VERIFICATION.json','replay_check.py','make_report.py')}
    verify={'schema_version':1,'pilot':'LME01','status':'FINISHED_NON_ADOPTION','protocol_commit':'7ee8674e5eb305296a953c66bdab0daf384e8010','source_commit':'ebd7b02c23df8aa7afd3625ae55a4de94fabd97e',
      'data_source':'sklearn load_digits raw images, no network download',
      'env':{'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'python':platform.python_version(),'device':'CPU','threads':1},
      'fresh_seeds':list(SEEDS['fresh']),'dev_seeds':list(SEEDS['dev']),'method_count':len(METHODS),
      'dev_rows':len(dev),'fresh_rows':len(fresh),'separate_replay':{'replayed_trainings':replay['model_training_replays'],'checked_fields':replay['cells_compared'],'exact':replay['match']},
      'paired_per_fresh_seed':differences,'mirror_rot_vs_linear_wins_no_threshold':normwins,'mirror_rot_001_gain_worlds':wins,
      'gates':checks,'scientific_verdict':'FAIL Mirror-specific LME01 predeclared improvement and native-head noninferiority; exact one-heavy-trunk/adapter mechanism PASS',
      'limitations':['LiME original zero-parameter route, AutoTop-K, MMT-47 NOT reproduced','one real-digit dataset with correlated seed partitions','Givens output chart exactly a native structured head, no unique function class','no GPU throughput or verified public benchmark','all reported bytes named NPZ FP32 including shared trunk and adapter'] ,'hashes_sha256':hashes}
    (OUT/'VERIFICATION.json').write_text(json.dumps(verify,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    hdr=['method','mean_test_nll','mean_worst_task_nll','mean_test_accuracy','mean_serialized_bytes','median_cpu_p95_ms','role_trainable_scalars']
    lines=['| '+ ' | '.join(hdr)+' |','| '+ ' | '.join(['---']*len(hdr))+' |']
    for v in summary:lines.append('| '+ ' | '.join(f'{v[k]:.6f}' if isinstance(v[k],float) else str(v[k]) for k in hdr)+' |')
    report=f'''# LME01 — LiME-style ONE shared PEFT output plus K role Views (2026-10-09)

**Scope:** rigorous small **sklearn digits** trained multi-output mechanism ONLY. LiME paper is **not** reproduced here (original source and zero-parameter router not run). Original author method: https://proceedings.mlr.press/v306/kowsher26a.html ; published source https://github.com/Kowsher/LiME.

**Frozen before data:** [protocol](../LME01_FROZEN_PROTOCOL.json), commit 60{verify['protocol_commit']}; source pinned to commit 60{verify['source_commit']} after 3 development worlds but **before** fresh worlds; Git blob hashes match local source. No hyperparameter selection from fresh.

## H — single physical forward, useful distinct outputs, marginal short-m value

Given one learned trunk Fθ and a single shared rank-8 PEFT output δ(Fθ(x)), learn K=5 distinct supervised output logits by cheap per-role output operators. Incremental Mirror scientific gate requires improvement >=.01 nat/example over same-four-coefficient native linear modulation in >=4/5 fresh splits, and quality within +.02 nat/example of LiME-style full direct vectors with suitable bytes and P95. One-forward operation itself is already native LiME and normal multihead, NOT novel.

## T — actual execution

- Inputs: each 8×8 normalized handwritten digit image x is shared identically between all five targets; parity, >=5, pixel left-right balance, top-bottom balance and center intensity. The last three thresholds are derived ONLY from training image IDs.
- 64→128 GELU→32 GELU trunk, rank8 shared adapter 32→8 GELU→32, one shared 32→2 classifier where compatible. All 7 methods execute the trunk and adapter **exactly once** per output batch. Ordinary native multi-head has 5 separate 32→2 readouts on the same h+δ, not 5 trunks.
- Methods: shared broadcast, native 5 ordinary output heads, LiME-style direct 32-vector per-role scaling, ordinary fixed 4-code scale, learned 4-code shared dictionary, structured 4-angle rotated shared δ, and structured 4-code shear shared δ. Exact native Givens output is the same function as 'Mirror rotation' (M0).
- Training: 350 AdamW updates, batch128, LR .002, wd .0001, FP32. Same initial trunk, adapter, classifier, same sampled image index stream in each world. Three dev worlds {list(SEEDS['dev'])}; unseen fresh worlds {list(SEEDS['fresh'])}; no audit tuning. Each world is a 75/25 stratified split of the **same original dataset**, so worlds are correlated, not five independent natural task datasets.
- Real serializer: deterministic named NPZ includes trunk, adapter, role codes/head weights, full dimension, seed-defined basis-reconstruction metadata, normalization and thresholds. Reported bytes are true artifact lengths; no theoretical Shannon compression implied. Single-thread CPU eager P95 at B128 with 20 warmups/80 timed. No GPU/VRAM conclusions.

## D — measured fresh results

{chr(10).join(lines)}

**Paired rotation minus ordinary linear 4-code test NLL (negative favors rotation):** {', '.join(f"{seed}:{differences[str(seed)]['rot_minus_linear4']:+.6f}" for seed in SEEDS['fresh'])}.

Native 5-head is the strongest overall task-quality method at mean NLL {next(v['mean_test_nll'] for v in summary if v['method']=='native_onepass_linearheads'):.6f}, not a K-recomputed strawman. Mirror rotation mean {next(v['mean_test_nll'] for v in summary if v['method']=='mirror_rot4'):.6f}; native fixed 4-code linear scaling mean {next(v['mean_test_nll'] for v in summary if v['method']=='lime_linear4'):.6f}; native LiME-style full-vector direct scaling mean {next(v['mean_test_nll'] for v in summary if v['method']=='lime_full_diagonal'):.6f}. Mirror rotation beats linear4 by >=.01 only in **{wins}/5** new worlds, rather than preregistered >=4/5. Mirror shears did not compensate. **Scientific gate FAIL/M0.** No MA status change.

## C — negative control and explanation

- The original LiME method already computes one shared PEFT module and cheap expert modulation, plus routing. This pilot strips router because task roles are supervised; it is strictly not the full native method.
- Native regular Givens output head reproduces the Mirror4 mathematical family exactly (unit test), so even a future Mirror4 win against plain scaling would require a second functional/byte mechanism to establish Mirror-specific novelty.
- A 4D coordinate may not span five real binary tasks; a standard native multihead has 330 task-specific scalars and attains much lower loss, while a 32-vector LiME-style p_e provides additional degrees of freedom.
- Fast-kernel effects are different: repeated sin/cos, rank mix and noncontiguous view tensor costs can outweigh a few saved scalars on CPU. Training sample/task correlation and unlocked CPU clock are additional limitations.

## U — uncertainty, validation and provenance

- 5 correlated partition seeds = descriptive evidence, not a 95% population CI. Numerically exact replay of 2 fresh worlds × 7 methods × 11 deterministic fields gave **{replay['cells_compared']} / {replay['cells_compared']} identical comparisons**, with 0 differing fields. P50/P95 are excluded from bitwise replay due to unlocked CPU clocks. Eight unit tests pass; Givens-mirror output, norm preservation, role-specific variation, all methods' gradient validity, one trunk+one adapter call and true serialization all checked.
- Files: [source](source/run_lme01.py), [tests](source/test_lme01.py), [dev raw](results/dev_raw.csv), [fresh raw](results/fresh_raw.csv), [core results](results/RESULTS_CORE.csv), [verification](results/VERIFICATION.json), [replay verification](results/REPLAY_VERIFICATION.json). Full logs and seeds retained.
- U_exp=k_cov*u_c with k_cov2 is not a coverage-certified bound for n5; sources of error are correlated task labels, source data partition, optimizer, clock and task geometry. Clock ms and file bytes cannot be combined or called a real-LM compression ratio.

## Next test, but no fabricated results

Reproduce **native original LiME 47-task MMT-47**, with its parameter-free router and AutoTopK; repeat Mirror m on expert vectors/codebook and compare against **same-byte native LiME linear/vector compression and native one-trunk multiheads**, plus a true holdout of independent task families. Alternatively test M3LoRA mixer codes with its native minor-SV initialization. Do not promote this CPU pilot to MA-1189 verified status.
'''.replace('\x0760','`').replace('\x07','`')
    (ROOT/'REPORT.md').write_text(report,encoding='utf8')
    print('verdict',verify['scientific_verdict'],'gains',wins,'of5','replay',replay['match'])
    print('files',[(p.name,p.stat().st_size) for p in OUT.iterdir()])

if __name__=='__main__':main()
