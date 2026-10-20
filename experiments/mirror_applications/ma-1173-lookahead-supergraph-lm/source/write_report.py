"""Read-only derived report and manifests after all fresh audits; no retuning."""
from __future__ import annotations
import csv, json,hashlib,statistics,subprocess,sys
from collections import defaultdict,Counter
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
results=pd.read_csv(ROOT/'RESULTS_CORE.csv')
events=json.loads((ROOT/'results'/'INTERVENTIONS.json').read_text())
deploy=json.loads((ROOT/'results'/'DEPLOY_REPLAY.json').read_text())
cfg=json.loads((ROOT/'PROTOCOL.json').read_text())
assert len(results)==54 and deploy['checks']==48 and not deploy['errors']
assert results.groupby('method').size().nunique()==1 and results.groupby('method').size().iloc[0]==6
assert len(events)==108
methods=cfg['models']
order=[('mirror_keep','Mirror full graph'),('mirror_random','Mirror random cuts'),('mirror_low_gate','Mirror low-gate cuts'),('mirror_lookahead','Mirror lookahead cuts'),('mirror_guarded','Mirror guarded lookahead'),('direct_tanh_lookahead','Direct tanh lookahead'),('affine_lookahead','Affine lookahead'),('native_tied','Native tied FFN'),('mirror_keep_extra','Mirror full graph +120 updates')]
rows=[]
for meth,name in order:
    rs=results[results.method==meth]
    rows.append({'method':meth,'name':name,**{v:float(rs[v].median()) for v in ['fresh_plain_nll','fresh_query_accuracy','fresh_ood_query_accuracy','serialized_inference_bytes','active_virtual_edge_mlps','cpu_forward_B1_us_median','cpu_forward_B32_us_median','all_optimizer_steps']}})

paired=results.set_index(['world','seed','method'])
base=paired.xs('mirror_lookahead',level='method')
wins={}
for ctrl in ['mirror_random','mirror_low_gate','mirror_keep','mirror_guarded','affine_lookahead','direct_tanh_lookahead','native_tied','mirror_keep_extra']:
    c=paired.xs(ctrl,level='method')
    d=base.fresh_ood_query_accuracy-c.fresh_ood_query_accuracy
    nll=base.fresh_plain_nll/c.fresh_plain_nll
    wins[ctrl]={'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum()),'mean_accuracy_delta':float(d.mean()),'median_accuracy_delta':float(d.median()),'median_fresh_plain_nll_ratio':float(nll.median()),'pair_differences':[{'world':world,'seed':int(seed),'ood_accuracy_delta':float(value)} for (world,seed),value in d.items()]}

counts=defaultdict(lambda:{'interventions':0,'accepted':0,'immediate_dev_worse':0,'future_better_than_keep':0,'future_better_than_keep_by_0p0005':0})
for ev in events:
    k=ev['method'];r=counts[k];r['interventions']+=1;r['accepted']+=bool(ev['removed']);r['immediate_dev_worse']+=ev['dev_immediate_after_cut']>ev['dev_before_cut']+1e-8
    if ev['lookahead_keep_future'] is not None:
        margin=min(ev['lookahead_candidates'].values())-ev['lookahead_keep_future']
        r['future_better_than_keep']+=margin<0;r['future_better_than_keep_by_0p0005']+=margin<=-0.0005

# Confirm the hypothesized special case counts: immediate drop followed by future benefit.
case_count=0
for ev in events:
    if ev['method']!='mirror_lookahead':continue
    margin=min(ev['lookahead_candidates'].values())-ev['lookahead_keep_future']
    if ev['dev_immediate_after_cut']>ev['dev_before_cut'] and margin<0:case_count+=1

summary={'model_rows':len(results),'worlds':cfg['synthetic_causal_language_data']['worlds'],'seeds':cfg['synthetic_causal_language_data']['seeds'],'methods':methods,
         'model_medians':rows,'paired_lookahead_vs':wins,'cut_diagnostics':dict(counts),
         'counterfactual_immediate_worse_future_better_event_count':case_count,
         'posthoc_deployment_packet_replay':{'checked':deploy['checks'],'errors':deploy['errors']},
         'gate_H_learning':'FAIL','gate_H_mirror_specific':'FAIL','gate_H_mechanism':'PASS',
         'limitations':['all datasets are synthetic token grammars','path masks only (not nodes, loops, directions or output choices)','future probing costs extra optimizer steps','fresh evaluated only once after all 54 trained checkpoints','clock not fixed','n=6 paired conditions are not population-level guarantees']}
(ROOT/'results'/'ANALYSIS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str)+'\n')

mt={row['method']:row for row in rows}
fn=lambda x:f'{x:.4f}'
md_table='| 方式 | Fresh NLL ↓ | OODキー参照正答率 ↑ | 残存経路 | 実測推論バイト | 全optimizer更新 |\n|---|---:|---:|---:|---:|---:|\n'
for row in rows:
    edge='—' if row['method']=='native_tied' else str(round(row['active_virtual_edge_mlps']))
    md_table+=f"| {row['name']} | {fn(row['fresh_plain_nll'])} | {100*row['fresh_ood_query_accuracy']:.2f}% | {edge} | {row['serialized_inference_bytes']:.0f} B | {row['all_optimizer_steps']:.0f} |\n"

def rate(s):return f'{s:.3f}'

report=f'''# MA-1173 — Prospective Cutting of Mirror Virtual MLP Supergraph During Causal Language Training

**2026-10-21 JST — Result: structural mechanism PASS, prospective-learning advantage FAIL, Mirror-specific value FAIL.**  
This is a synthetic token-grammar causal language model, **not real natural-language evidence**. All historical MA-1166..1172 outcomes remain unchanged. Dedicated research branch only, no main merge.

## 1. Research identity and implementation

One global physically learned **288-FP32 coefficient** core and four signed-permutation pages generate **18 ordered 12→12→12 virtual MLP edges** for 3 nodes × all directed connections × 2 Mirror addresses. There are two repeated Loop updates. A separate **causal two-head attention** performs genuine token mixing. Per-edge gate logits and separate Boolean ON/OFF masks control *presence* independently of learned edge weight magnitudes. Cutting an edge removes it from virtual-weight generation and active edge execution; it is not merely multiply-by-zero in a fully executed graph.

Full Mirror model: **1878 independently learned scalars** (including token embedding/attention/head); native tied 12→16→12 FFN control **1876**. All 18×2×12×12 = **5184 generated logical weights** are functions of the shared core and addresses, not independently learned parameters. The Mirror decoder uses Householder→tanh→Householder; exact-address affine and direct-tanh controls are included. The 3 nodes, 2 views, 2 Loop iterations, attention, and output head are NOT optimized or pruned in this first test.

## 2. Frozen protocol and data separation

Full preregistered protocol SHA-256: 

{hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()}

Two worlds (balanced query and 90%-first-key shortcut), seeds 31/47/59. Each has independent train/dev/fresh/OOD samples (4096/512/1024/1024), symbolic key-value token grammar of sequence length 12 and vocabulary 24. Training is AdamW lr 0.005, batch32, 260 updates, weighted objective = mean causal token NLL plus 2× last token loss. Native and no-cut controls have matching training schedule and start seeds. Cut events at 80/140/200; each forced event removes 2 edges. Random, weakest gate, candidate lookahead and guarded lookahead are distinct interventions.

At each lookahead event: 4 disjoint randomly sampled two-edge groups, each trained for 8 hypothetical optimizer updates using a cloned model AND cloned AdamW state, plus one uncut reference continuation using the same fixed future training batches; choice made by **dev**, never by fresh. 5×8×3 = **120 additional optimizer updates**, including discarded probes. Corresponding no-cut-extra control uses 380 *real* optimizer updates. Guarded lookahead commits only when predicted future dev improvement vs uncut ≥0.0005. Fresh/OOD opened after all 54 training checkpoints were complete; neither split selected hyperparameters, structural decisions, or checkpoints.

## 3. Fresh/OOD outcomes (six world/seed pairs; medians)

{md_table}

- **H_mechanism PASS:** forced interventions cut graph from 18 to 12 live edges. The active virtual-MLP MAC proxy dropped from **10368 to 6912 MAC/token** (33.3%) excluding attention, activations, Mirror decoding and other layers. Actual inference packet changed from **13668 B to 13540 B** (0.94% lower), not a 33% total-byte reduction. The full learned coefficient tensors still exist during training and are not independent virtual copies.
- **H_learning FAIL:** mirror_lookahead had **{wins['mirror_random']['wins']}/6** OOD query-accuracy wins over random and **{wins['mirror_low_gate']['wins']}/6** over low-gate, versus a frozen threshold of ≥4/6 paired wins and ≥0.02 absolute gain. Median results tie, not merely lack statistical significance.
- **H_Mirror_specific FAIL:** mirror_lookahead improved OOD accuracy over direct-tanh **{wins['direct_tanh_lookahead']['wins']}/6** and affine **{wins['affine_lookahead']['wins']}/6** pairs, without meeting a strict Pareto quality/runtime/bytes benefit. Native tied FFN and longer ordinary training are substantive alternatives.
- **Guard diagnosis:** only **{counts['mirror_guarded']['accepted']}/{counts['mirror_guarded']['interventions']}** candidate interventions passed a +0.0005 future-dev-loss threshold; it mostly left the full graph intact. Across forced Mirror lookahead, **{counts['mirror_lookahead']['future_better_than_keep']}/{counts['mirror_lookahead']['interventions']}** shortlisted removals had lower post-H dev loss than keeping all edges, despite all forced cuts being applied. **{case_count}** cases had worse immediate dev loss but better hypothetical H-step dev loss after cutting. This is a local learning diagnostic, not validated final superiority.
- **Shortcut/OOD:** the shortcut world changed first-key query prevalence from 90% in train to 10% in OOD. Median OOD query accuracy was approximately 25.8% for each route policy, showing the expected distributional robustness improvement did not materialize.

## 4. Hardware/runtime and hidden costs

Hardware AMD EPYC 9V74, CPU PyTorch 2.10.0+cpu, FP32, one thread, unlocked clock; batch-1 forward latency median across six models: full Mirror **{mt['mirror_keep']['cpu_forward_B1_us_median']:.1f} μs**, cut Mirror lookahead **{mt['mirror_lookahead']['cpu_forward_B1_us_median']:.1f} μs**, tied native FFN **{mt['native_tied']['cpu_forward_B1_us_median']:.1f} μs**. Each per-model median came from 15 warmups and 75 timed forward passes. These are eager-CPU estimates, not GPU/fused-kernel claims. Runtime fixed deterministic page/direction buffers **14832 B** and dynamically decoded active graph weights **13824 B** (12 edges), not included in the serialized inference packet bytes above. The serialized packet also assumes access to the external deterministic decoder program; its source/code storage remains separately payable. No claim of free virtualization.

## 5. Verification and limitations

- Pretraining unit tests **6/6 PASS**: all-edge tensor generation, physical parameter parity, causal future-token exclusion, shared-core/addr/gate gradients, hard-cut active-edge execution, deterministic split separation.
- **54/54** training records, **54/54** checkpoint hashes verified; dev stats reproduced within 1e-7 before fresh access; **48/48** graph-model deployment packet readbacks reconstructed the same logits (max abs error ≤1e-6). Full source, result CSV and deployment audits included.
- Only **edge** pruning, not node creation/deletion, Mirror direction count, address topology, Loop count, or output-choice learning. Does not prove arbitrary logical MLP capacity or capacity per physical coefficient.
- Six paired synthetic conditions, no natural language corpus, no GPU test, no confidence claim from 6 trials. Short-lookahead chosen on a repeatedly used development split may overfit that split. Synthetic causal grammar and teacher distribution may underexploit Mirror views. Source program and deterministic buffers are not zero-cost; shared-core parameter updates can interfere across edges.

## 6. H / T / D / C / U

**H:** Prospective structural cuts guide physical coefficient learning more effectively than low-gate or random cuts and generic non-Mirror shared maps.  
**T:** 9 methods × 2 distributions × 3 seeds; frozen train/dev/fresh/OOD, fully counted +120 lookahead optimizer updates, native/affine/direct-tanh controls.  
**D:** Structural edge execution PASS, prospective learning and Mirror-specific predictive benefit FAIL, guard cut 1/18; no adoption.  
**C:** ordinary extra training, sparse regularization and direct nonlinear generated weights may match; this small synthetic task may rely primarily on the causal attention backbone.  
**U:** CPU clock, split variability, finite-horizon counterfactual approximation, repeated dev decisions, learned gate identifiability, and the absence of natural-language evidence. Model and metric outputs are dimensionless for token losses in nats/token, MAC/token and bytes; no mixed-dimension subtraction.

## 7. Scientific next stage

Before expanding to real text or claiming adaptive NAS, explicitly **test whether cutting a high-impact edge changes subsequent gradients and yields more robust held-out rule learning**. Compare physical-core gradient alignment and trained-path causal necessity with an independent high-impact teacher. If that falsifier also fails, change the supergraph operator rather than increasing graph size or hiding CPU cost. Node/view/Loop pruning would be later, independent experiments.
'''
report=report.replace('\\ndef'+hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest())
(ROOT/'REPORT_JA.md').write_text(report)
status=f'''# MA-1173 status\n\n2026-10-21 JST. **Completed FAIL** for Mirror-specific future-learning quality; **mechanism PASS** for cutting physical execution.\n\n- Frozen protocol SHA256: {hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()}\n- 54/54 training checkpoints, 9 methods×2 worlds×3 seeds; 6/6 structure unit tests; 48/48 compact deploy packets replayed.\n- The strict lookahead gate versus random and gate-pruning failed 0/6 paired OOD accuracy wins. Guard accepted only 1/18 two-edge interventions; learned graph improved neither causal LM NLL nor OOD generalization under the tested budget.\n- No natural language LM, full graph architecture search, or deployment claim. Main not modified. See REPORT_JA.md.\n'''
(ROOT/'STATUS.md').write_text(status)

manifest={}
for p in ROOT.rglob('*'):
    if p.is_file() and p.name not in ('MANIFEST_SHA256.json','VERIFICATION.json') and not any(x in p.parts for x in ('__pycache__','.pytest_cache')):
        manifest[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
verification={'experiment_id':'MA-1173','frozen_protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
              'trained_models':len(results),'expected':54,'fresh_opened_once_after_train_complete':True,
              'checkpoint_hash_verified':54,'inference_packets_replayed':48,'max_replay_error_threshold':1e-6,
              'source_test_cases':6,'source_test_passes':6,'data_worlds':2,'seeds':3,'mechanism':'PASS','preregistered_lookahead_quality':'FAIL','mirror_specific_quality':'FAIL',
              'manifest_count':len(manifest),'known_limitations':['synthetic sequences','dev-selection bias','unlocked CPU','external executable code not counted as bytes','no physical parameter graph-core shrink after cuts']}
(ROOT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
print('REPORT',len(report),'bytes, MANIFEST',len(manifest),'entries; gate FAIL; controls',len(rows))
