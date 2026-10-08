from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json,csv,hashlib,platform
import torch
base=Path(__file__).resolve().parent
r=base/'results'
sets={'development':pd.read_csv(r/'dev_raw.csv'),
      'initial_fresh':pd.read_csv(r/'fresh_raw.csv'),
      'postdiscovery_replication':pd.read_csv(r/'replica/fresh_raw.csv')}
expected={'development':(3,[31,32,33]),'initial_fresh':(5,[301,302,303,304,305]),'postdiscovery_replication':(5,[401,402,403,404,405])}
methods=('shared','native_multihead','native_fastweight','mirror_hadamard4','native_hadamard_linear4','native_direct_diagonal4','native_learned_dictionary4','mirror_private1','native_linear_private1')
flags={};records=[]
for stage,data in sets.items():
    count,seeds=expected[stage]
    assert len(data)==count*len(methods)
    assert sorted(data.seed.unique().tolist())==seeds
    assert set(data.method)==set(methods)
    assert (data.trunk_invocations_per_forward==1).all()
    assert (data.K==5).all()
    assert data.input_same_across_roles.all()
    assert (data.training_steps==400).all()
    assert (np.isfinite(data[['fresh_mean_nll','fresh_accuracy','fresh_worst_task_nll','serialized_npz_bytes','cpu_p95_ms']].to_numpy()).all())
    assert len(data.drop_duplicates(['seed','method']))==len(data)
    for method,q in data.groupby('method'):
        records.append(dict(stage=stage,method=method,worlds=q.seed.nunique(),
                            NLL_mean=q.fresh_mean_nll.mean(),accuracy_mean=q.fresh_accuracy.mean(),
                            worst_task_nll_mean=q.fresh_worst_task_nll.mean(),total_npz_bytes=q.serialized_npz_bytes.mean(),
                            member_scalars=q.member_scalars.mean(),cpu_p95_ms_mean=q.cpu_p95_ms.mean(),
                            trunk_calls=1,role_output_std=q.out_prob_std.mean()))
    flags[stage]={'seeds':seeds,'rows':len(data),'positive_binary_tasks':True,'exact_single_trunk_call':True,
                 'no_fresh_tuning':True,'dataset_natural_worlds':'same sklearn digits, overlapping splits'}
core=r/'RESULTS_CORE.csv'
with core.open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(sorted(records,key=lambda x:(x['stage'],x['NLL_mean'])))
initial=sets['initial_fresh'].set_index(['seed','method']).fresh_mean_nll.unstack()
repl=sets['postdiscovery_replication'].set_index(['seed','method']).fresh_mean_nll.unstack()
paired_initial=(initial.mirror_hadamard4-initial.native_hadamard_linear4)
paired_repl=(repl.mirror_hadamard4-repl.native_hadamard_linear4)
paired_private=(repl.mirror_private1-repl.native_linear_private1)
flags['hypothesis']={
  'initial_mirror_minus_linear_mean':float(paired_initial.mean()),
  'initial_mirror_better_worlds':int((paired_initial<0).sum()),
  'replica_mirror_minus_linear_mean':float(paired_repl.mean()),
  'replica_mirror_better_worlds':int((paired_repl<0).sum()),
  'replica_margin_le_minus_0p01':int((paired_repl<=-.01).sum()),
  'replica_margin_gate_pass':bool((paired_repl<=-.01).sum()>=4),
  'replica_mirror_private_minus_linear_private_mean':float(paired_private.mean()),
  'native_5head_inference_true_forward_calls':1,
  'Mirror_specific_adoption':False}
flags['freeze']={
 'MAT03_protocol_commit':'52afafbae5b4f1888e632488f85c430e01d749fa',
 'MAT03_main_source_frozen_before_replica_commit':'825c281dbdcda67168a3c98289a7cc04a802f631',
 'MAT03R_protocol_and_driver_frozen_commit':'c6a1a16974a607b51fc121f4ca57f75fc634942b',
 'MAT03_initial_source_not_remote_pre_fresh':'True: initial frozen design was committed but full implementation was not committed to GitHub until after the 301..305 exploratory run. The independent 401..405 replication has the exact source pinned beforehand.',
 'source_sha256_original':'e63b0f6cdc1f66d50cdd00629c05b4d0b800b73b9d24095a24076cad3736ffbe'}
flags['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'bench_cpu_only':True,'CUDA_tested':False}
flags['limits']=['5 fresh are repeated splits of one small dataset (not new independent task families)','Hadamard-exp View is a structured feature gate related to IA3/FiLM, not new one-forward invention','same-code linear chart is mandatory strong null','native one-trunk multihead is already one call','No LM token loss, MoE native reproduction, GPU offload']
flags['source_sha256']={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'source').glob('*.py'))}
flags['result_sha256']={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((r).rglob('*.csv'))}
flags['replay']=json.loads((r/'REPLAY_VERIFICATION.json').read_text())['all_pass']
assert flags['replay']
(r/'VERIFICATION.json').write_text(json.dumps(flags,indent=2,sort_keys=True)+'\n')

def table(stage):
    data=sets[stage]
    group=data.groupby('method').agg(loss=('fresh_mean_nll','mean'),accuracy=('fresh_accuracy','mean'),bytes=('serialized_npz_bytes','mean'),P95=('cpu_p95_ms','mean'),worst=('fresh_worst_task_nll','mean'))
    o=['native_multihead','mirror_private1','native_linear_private1','native_fastweight','mirror_hadamard4','native_hadamard_linear4','native_direct_diagonal4','native_learned_dictionary4','shared']
    line=['| method | mean NLL ↓ | accuracy ↑ | whole NPZ B ↓ | CPU p95 ms ↓ |','|---|---:|---:|---:|---:|']
    for name in o:
        v=group.loc[name]
        line.append(f'| {name} | {v.loss:.6f} | {v.accuracy:.4f} | {v.bytes:.0f} | {v.P95:.4f} |')
    return '\n'.join(line)

report=f'''# MAT03 / MAT03R — genuine same-input, one-heavy-forward, five-useful-output evidence

**2026-10-09 CPU experiment report. Scope: sklearn handwritten digits, five binary tasks on exactly the same original input image.** This repairs the input-transformation confound in MAT01/MAT02. It is NOT a natural language model, native MIMO/TabM/LoGo paper replication, GPU study, independent biological/linguistic function capacity theorem or Mirror-specific adoption result.

## Task, method, and evidence firewall

- Every original 8×8 image is processed **once** by the same trainable shared trunk 64→128 GELU→32 GELU; a forward hook verified **exactly one trunk forward** per batch across all nine methods.
- Each input has **five supervised binary outputs**: digit parity, digit >=5, horizontal pixel-mass imbalance, vertical mass imbalance, center pixel density. The latter three use thresholds estimated **only from the training images**, not audit images. All methods see identical normalized input pixels and all five labels. The five tasks are useful for this synthetic/observed classification fixture but do not prove independent generative skills or language capabilities.
- **Native baselines:** ordinary five linear output heads on one trunk, shared output head with zero role conditioning, BatchEnsemble/IA3-like per-role 32D gate, four-coefficient fixed/original feature diagonal gate, ordinary fixed-Hadamard affine m basis, learned linear 4×32 dictionary m bank, and rank1 paid private corrections. The stronger native one-trunk five-head baseline is NOT charged five heavy forward passes.
- **Mirror m:** a 4D task code controls \\(Q_m=H\\operatorname{{diag}}(\\exp(Bm))H^T\\),\\ where H is a deterministic normalized Hadamard32 and B is a deterministic 32×4 source-independent mixing basis; the common final classifier is shared. The sole View-specific work is small-code scaling and 32D readout after the one heavy trunk call. **This is mathematically a structured feature modulation**, closely related to established IA3/FiLM; it must beat an ordinary same-byte affine role code before calling its geometry useful.
- **Fair training:** identical trunk initialization by world seed, 400 AdamW updates and 128-input batches/method, same data/update sampling per method, one-thread CPU. Five tasks trained end-to-end (not an oracle-provided common intermediate). Small rank1 private heads in two conditions are fully paid.
- **Serializer:** actual deterministic named-NPZ inference artifact, including trunk, task-specific readouts/codes, training-only preprocessing and task-threshold/config metadata; the fixed Hadamard/mixing generators use declared deterministic algorithm + seed. Byte differences on whole model are naturally modest because trunk dominates.

## Initial MAT03 [301..305] — exploratory / initial registered protocol

- Initial MA protocol preregistered at GitHub commit `52afafbae5b4f1888e632488f85c430e01d749fa` before task data audit.
- The actual MAT03 source was **not remotely committed before this initial fresh evaluation**. Do not describe its 301..305 evidence as fully code-frozen independent replication. The source was subsequently pinned for MAT03R.
- 3 development worlds [31,32,33], 5 initial fresh worlds [301..305]; nine complete methods per seed, all five task targets.

{table('initial_fresh')}

In the initial five fresh repeated partitions, Mirror4 vs same-code linear4 NLL differences (Mirror minus linear, negative favors Mirror): **{', '.join(f'{v:+.5f}' for v in paired_initial)}**; 5/5 negative. Mirror4 NLL ~0.125 vs linear4 ~0.149, but native full one-trunk five heads reaches ~0.071 and still executes one trunk call. Mirror-only saves only ~1.35% **whole inference NPZ bytes** (not 10%) relative to full five-head native and misses its task quality. Mirror+private rank1 reaches ~0.073 NLL, but requires **more total bytes** and more inference time than native five-head. This does not establish a Mirror-specific Pareto advantage.

## Independently preregistered post-discovery replication MAT03R [401..405]

- After seeing the above effect, pin the exact execution source SHA256 \\`e63b0f6cdc1f66d50cdd00629c05b4d0b800b73b9d24095a24076cad3736ffbe\\` at commit \\`825c281dbdcda67168a3c98289a7cc04a802f631\\`.
- Freeze new replication seeds [401..405], unchanged source, same 400 steps and all nine baseline methods, and an explicit **>=0.01 nat/example Mirror gain over linear4 in >=4/5 fresh worlds** at pre-audit commit \\`c6a1a16974a607b51fc121f4ca57f75fc634942b\\`. No new dev optimization on these seeds.

{table('postdiscovery_replication')}

Paired Mirror4 minus affine linear4 NLL by fresh world **{', '.join(f'{v:+.5f}' for v in paired_repl)}**. Mirror has the lower error in 4/5 worlds but **only 1/5** meets the frozen 0.01 nat/label threshold. Mean improvement is just **{-paired_repl.mean():.6f}** nat/example, with one reversal (+0.02256). Therefore **MAT03R H1 gate FAIL**.

- Native multihead mean NLL **{sets['postdiscovery_replication'].query("method == 'native_multihead'").fresh_mean_nll.mean():.6f}** vs Mirror4 **{sets['postdiscovery_replication'].query("method == 'mirror_hadamard4'").fresh_mean_nll.mean():.6f}**: no quality noninferiority. Whole NPZ around 53.5–54.2 kB gives only ~1.35% savings for Mirror4 versus native five heads. **H2 FAIL.**
- Mirror+private rank1 is closer in loss but needs more bytes and higher P95 than native five-head. On MAT03R it is slightly **worse than the ordinary same-code linear+private1 in 5/5 fresh worlds**, mean paired difference +{paired_private.mean():.6f} nat/label. **H3 FAIL.**
- Output/label diversity and single shared trunk are genuine and checked in 10 shape/data/gradient unit tests; **H4 PASS mechanical only**. The five tasks are binary classification on one fixed real-image dataset, not multiple independent LLM expert functions.

## Decision, counterexplanations and ERROR CHECK

**Strong conclusion:** Useful multi-output inference from one shared expensive trunk is possible and already realized by all the successful native methods too. A short structured 4D Mirror chart improves over some extremely small affine/diagonal controls in the first five splits, but independent frozen-code replication does **not** support the preregistered margin. Native ordinary multihead is both more accurate and much faster, at modest whole-model storage increase. No Mirror-specific Pareto superiority is demonstrated; the full MA-1175 scientific status remains UNTESTED.

- Different outputs do not automatically imply new Shannon information or independent model capacity. A fixed invertible Hadamard chart may be folded into a standard head for inference; exp feature gates are an established nonlinearity.
- Task overlap: 10 experimental fresh partitions are **not 10 independent datasets**, and the two sets are sequential hypothesis discovery/replication, so do not pool them into a fake first-time trial. Report paired seed differences rather than treating 5×450 image labels as independent observations.
- Real hardware caveat: this is eager CPU PyTorch 2.10 FP32 with unlocked clocks, batch128, warmup20 and 100 timed calls. No CUDA/GPU transfer/VRAM/native paper throughput extrapolation. Full file bytes are physically serialized NPZ; packed/fused/compiled code could alter runtime ranking.
- The full source was frozen on GitHub before MAT03R fresh. Six repeat train/eval cells (3 native vs Mirror/linear for seed301, three for seed401) replayed with **zero difference in deterministic numeric fields**. Ten original MATLAB-like mechanism unit tests pass; historical MAT01/MAT02 tests remain intact.
- U: five overlapping data splits, optimizer stochasticity fixed, early-phase source code freeze distinction, possible repeated-dataset dependence, surrogate physical latency, numerical precision, task thresholds estimated from training data. No coverage-calibrated population 95% CI from n=5. For task loss L in nat/label, \\(u_c^2=u_{{seed}}^2+u_{{test}}^2+u_{{numeric}}^2\\) only under independence; include covariance otherwise. k=2 expanded uncertainty is indicative rather than guaranteed.

## Reproduce

- [Pre-audit protocol for initial study](../MAT03_FROZEN_PROTOCOL.json), [source-freeze and new-seed replication protocol](MAT03R_FROZEN_PROTOCOL.json).
- [Frozen PyTorch implementation](source/run_mat03.py), [ten tests](source/test_mat03.py), [repetition runner](source/run_mat03_replica.py).
- [Initial dev/fresh and independent replication CSVs](results/), [compact results](results/RESULTS_CORE.csv), [hashes and numerical gates](results/VERIFICATION.json), [six-cell exact replay](results/REPLAY_VERIFICATION.json).
- No main, worker queue, other MA status or original science claim was edited by these experiments.
'''
report=report.replace('\\`','`')
(base/'REPORT.md').write_text(report)
print(json.dumps({'development_rows':len(sets['development']),'initial_rows':len(sets['initial_fresh']),'replica_rows':len(sets['postdiscovery_replication']),'replica_win_4_of_5':flags['hypothesis']['replica_mirror_better_worlds'],'replica_margin_pass':flags['hypothesis']['replica_margin_gate_pass'],'report_chars':len(report)}))
