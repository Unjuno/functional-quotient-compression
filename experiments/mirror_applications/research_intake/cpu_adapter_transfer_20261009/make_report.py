from pathlib import Path
from hashlib import sha256
import pandas as pd
import numpy as np
import json
import csv
import sys
import platform
import torch

base=Path(__file__).resolve().parent
root=base/'results'
series=['mat01','mat02']
rows=[]
checks={}
for series_id in series:
    folder=root/series_id
    dev=pd.read_csv(folder/'dev_raw.csv')
    fresh=pd.read_csv(folder/'fresh_raw.csv')
    methods=set(fresh.method)
    seeds=([11,12,13] if series_id=='mat01' else [21,22,23],[101,102,103,104,105] if series_id=='mat01' else [201,202,203,204,205])
    checks[series_id]={
        'dev_rows':len(dev),'fresh_rows':len(fresh),
        'dev_seeds':sorted(int(x) for x in dev.seed.unique()),'fresh_seeds':sorted(int(x) for x in fresh.seed.unique()),
        'methods':sorted(methods),
        'roles':sorted(fresh.role.unique().tolist()),
        'unique_seed_method_role':len(fresh)==len(fresh[['seed','method','role']].drop_duplicates()),
        'same_source_basis_within_seed':all(fresh.groupby('seed').source_family_sha256.nunique()==1) if 'source_family_sha256' in fresh else all(fresh.groupby('seed').source_svd_sha256.nunique()==1),
        'true_dataset_hash_consistent':len(set(fresh.base_dataset_sha256 if 'base_dataset_sha256' in fresh else fresh.dataset_sha256))==1,
        'all_values_finite':bool(np.isfinite(fresh[['nll','accuracy','serialized_npz_bytes','cpu_p50_ms','cpu_p95_ms']]).to_numpy().all()),
        'NPZ_physical_bytes_positive':bool((fresh.serialized_npz_bytes>1000).all()),
        'no_repeated_physical_copy_rows':all(fresh.groupby(['seed','method']).serialized_npz_bytes.nunique()==1)
    }
    expected=24 if series_id=='mat01' else 32
    assert len(dev)==3*expected and len(fresh)==5*expected,checks
    assert checks[series_id]['unique_seed_method_role'] and checks[series_id]['same_source_basis_within_seed']
    assert checks[series_id]['true_dataset_hash_consistent'] and checks[series_id]['all_values_finite']
    if series_id=='mat01':
        expected_names={'base','native_lora4','mirror4','linear4','diag4','fullcore16'}
    else:expected_names={'base','native_lora4','native_lora2','mirror4','linear4','mirror4_private2','linear4_private2','fullcore16_private1'}
    assert methods==expected_names
    for meth,part in fresh.groupby('method'):
        rec={'series':series_id,'method':meth,'world_count':part.seed.nunique(),'role_count':part.role.nunique(),
        'fresh_nll_mean':float(part.nll.mean()),'fresh_accuracy_mean':float(part.accuracy.mean()),
        'npz_bytes_mean':float(part.serialized_npz_bytes.mean()),'cpu_p50_ms_mean':float(part.cpu_p50_ms.mean()),
        'cpu_p95_ms_mean':float(part.cpu_p95_ms.mean())}
        rows.append(rec)
    s=fresh.groupby(['seed','method']).nll.mean().unstack()
    checks[series_id]['matched_world_nll']={str(int(seed)):{k:float(v) for k,v in row.dropna().items()} for seed,row in s.iterrows()}

core=root/'RESULTS_CORE.csv'
with core.open('w',encoding='utf-8',newline='') as fp:
    wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(sorted(rows,key=lambda x:(x['series'],x['fresh_nll_mean'])))

checks['original_unit_tests']={'MAT01':9,'MAT02':5,'prior_batch_regression':17}
checks['limitations']={
    'native_LoGo_reproduced':False,
    'natural_LLM_or_GPU_benchmark':False,
    'independent_natural_dataset_worlds':False,
    'shared_inference_runtime_uses_same_canonical_input_per_role':'Yes: this measures a shared-trunk readout kernel, not inference for four different corrupted image inputs; per-role transformed images require distinct encoder work.'
}
checks['source_hashes']={str(p.relative_to(base)):sha256(p.read_bytes()).hexdigest() for p in sorted((base/'source').glob('*.py'))}
checks['results_hashes']={str(p.relative_to(base)):sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('**/*.csv'))}
checks['protocol_commits']={'MAT01':'5d4c37a1ba70b162fa09bbe2182a4e2d206f5eac','MAT02':'c7b7b517c66d757930baf4b7054535c13a0dcf2a'}
checks['dtype']='torch float32, SVD float64, NPZ float32 arrays'
checks['environment']={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'platform':platform.platform()}
verify=root/'VERIFICATION.json'
verify.write_text(json.dumps(checks,indent=2,sort_keys=True)+'\n')

mat01=pd.read_csv(root/'mat01/fresh_raw.csv').groupby('method').agg(nll=('nll','mean'),acc=('accuracy','mean'),bytes=('serialized_npz_bytes','mean'),p95=('cpu_p95_ms','mean'))
mat02=pd.read_csv(root/'mat02/fresh_raw.csv').groupby('method').agg(nll=('nll','mean'),acc=('accuracy','mean'),bytes=('serialized_npz_bytes','mean'),p95=('cpu_p95_ms','mean'))
mat02w=pd.read_csv(root/'mat02/fresh_raw.csv').groupby(['seed','method']).nll.mean().unstack()

fmt=lambda x:f'{x:.5f}'
text=f'''# MAT01 / MAT02 — CPU adapter-bank Mirror falsification and private-residual follow-up

**Research-only observational report (2026-10-09).** Two independently preregistered CPU experiment series on sklearn's 8x8 handwritten digit images; they are NOT a reproduction of the LoGo method nor a real LLM. The full original MA-1178 stays **UNTESTED**. We ran the negative results rather than claiming useful compression.

## Frozen order, identity and evidence

- MAT01 protocol GitHub commit `5d4c37a1ba70b162fa09bbe2182a4e2d206f5eac`: dev [11,12,13], fresh [101..105].
- MAT02 pre-fresh amendment commit `c7b7b517c66d757930baf4b7054535c13a0dcf2a`: dev [21,22,23], fresh [201..205]. It was created **after** MAT01's negative result. No retroactive alteration of MAT01.
- Dataset sklearn `load_digits`, 1,797 images, original IDs stratified train 75%/audit 25%; normalization learned from training images only. All target conditions use the **same 300 original support image IDs** within a seed and label set, with 4 role changes: diagonal shifts, Gaussian additive noise, pixel dropout. Six independent source LoRAs train on horizontal/vertical shift, blur and contrast source roles.
- One source-trained 64-hidden GELU base classifier; after freezing, source role-specific rank-4 LoRAs are trained. Derive source-only U[64,4], V[4,10], mean core C[4,4] and source PCA core dictionary (4 atoms of 4x4). For unknown target transformations, train 4D Mirror angles, 4D plain linear coefficients, native independent rank4/rank2, full 16D core and/or rank1/rank2 private deltas on the same target support and update budget.
- All task losses are **heldout fresh per-original-image classification cross entropy (nat/example)** and accuracy. Actual named NPZ model bytes include base, preprocessing, paid task-bank representation and metadata. The heldout dataset ID set remains unseen by every optimiser.
- Fixed CPU PyTorch 2.10.0+cpu, sklearn 1.8, NumPy 2.3.5, torch+BLAS threads1, B128, 20 warmups /100 calls per method (virtual EPYC CPU, unlocked clock). Native LoGo dynamic router/adapter merging is **not implemented**.

## A. MAT01 — source-aligned low-dimensional codes vs true native low-rank adaptation

**fresh 5 dataset partitions × 4 roles = 120 per-role rows**:

| model | NLL ↓ | Accuracy ↑ | actual total NPZ byte ↓ | CPU P95 ms ↓ |
|---|---:|---:|---:|---:|
'''
for name in ['native_lora4','fullcore16','linear4','diag4','mirror4','base']:
 r=mat01.loc[name];text+=f"| {name} | {fmt(r.nll)} | {fmt(r.acc)} | {int(round(r.bytes))} | {fmt(r.p95)} |\n"
text+='''
Native rank4 LoRA wins decisively on task performance. The short Mirror code is weaker than the same-dimensional ordinary source-trained linear code on all five independent seeds; a 16D unrestricted shared-core model still cannot match native LoRA. The source learned 4D left/right subspaces do not capture the transformations needed for heldout target-image shift functions. **MAT01 scientific mechanism gate FAIL.**

## B. MAT02 — source Mirror or linear code plus genuinely private rank2 residual

**Separate fresh 5 dataset partitions × 4 roles = 160 per-role rows**, with independent fresh seeds [201..205]. A follow-up created after MAT01's visible dev/fresh failure; not pooled as one preregistered experiment.

| model | NLL ↓ | Accuracy ↑ | actual total NPZ byte ↓ | CPU P95 ms ↓ |
|---|---:|---:|---:|---:|
'''
for name in ['native_lora4','linear4_private2','mirror4_private2','native_lora2','fullcore16_private1','linear4','mirror4','base']:
 r=mat02.loc[name];text+=f"| {name} | {fmt(r.nll)} | {fmt(r.acc)} | {int(round(r.bytes))} | {fmt(r.p95)} |\n"
text+=f'''
The rank2 private residual brings Mirror NLL **{fmt(mat02.loc['mirror4'].nll)} → {fmt(mat02.loc['mirror4_private2'].nll)}**, demonstrating the private residual carried most needed function variation. But **Mirror+private2 ({fmt(mat02.loc['mirror4_private2'].nll)}) remains inferior to ordinary linear4+private2 ({fmt(mat02.loc['linear4_private2'].nll)})** and independent rank4 LoRA ({fmt(mat02.loc['native_lora4'].nll)}). In only 1/5 fresh worlds does Mirror+private2 beat its plain linear+private2 counterpart. Paired Mirror minus native NLL by fresh world: {', '.join(fmt(v) for v in (mat02w.mirror4_private2-mat02w.native_lora4))}. Mirror+private2 takes {int(round(mat02.loc['mirror4_private2'].bytes))} physical persisted NPZ bytes versus native rank4 LoRA {int(round(mat02.loc['native_lora4'].bytes))} — **larger, not smaller**. This method is slower in the eager CPU implementation because each Givens core is reconstructed by Python/PyTorch operations.

**MAT02 preregistered mechanism gate FAIL**: quality, persisted bytes, latency, and Mirror-specific non-Mirror paired comparison do not all pass. The logical function isn't regained by free coordinate views; it required paid private capacity. This is not a theorem excluding other Mirror coordinate families.

## Critical truthfulness caveat: no single heavy pass for four different image shifts

Each real task uses **different transformed pixels before the frozen encoder**. Therefore the recorded CPU benchmark (one shared backbone on an identical canonical unshifted input followed by four role readouts) is a **synthetic identical-input readout kernel diagnostic**, not the actual per-role deployment workload. In real multitask evaluation with four differently corrupted input images, the feature encoder must process each distinct image (unless a separately trained canonicalization/shared sufficient statistic exists). Do not claim a real 4-role one-forward speedup from these timings; the quality and role accuracies use separately transformed, correctly heldout inputs, while the runtime fixture uses a common dummy input. Matching native controls were timed on the same dummy input, so relative decoder overhead remains an implementation diagnostic only.

**GPU, true LoGo algorithm, natural language, real multi-output same-image task, and independent dataset-world replication are all NOT DONE.** Despite the feature-input mismatching benchmark caveat, the functional quality negative results on shifted digits remain valid for that limited conditional task.

## Verification and decision

- MAT01 dev: 72 condition-role rows, fresh:120; MAT02 dev:96, fresh:160. Distinct world seed sets; no missing method/role/seed cell. All per-seed source subspace hashes and dataset hashes consistent, all metrics finite.
- MAT01 9 original unit tests and MAT02 5 original tests passed, as did all 17 prior batch unit tests (same-session only). Checkpoint weights were not stored; deterministic seeded retrain from original immutable source is the replay route.
- MAT01 scientific pilot **FAIL** (no Mirror-specific beneficial Pareto), MAT02 **FAIL** (private improves but simple native/linear remains better). Parent MA-1178 **UNTESTED** until real LoGo and natural task quality/performance are reproduced.
- Confounds: source rank4; extreme unseen diagonal input shifts; fixed 320/145/175 optimizer steps; role-specific target train data; actual serializer metadata; unloaded vs resident task buffers; virtual CPU kernel overhead; five correlated partitions from the same 1,797 images.
- Useful next experiment: compare source rank R=8/16 vs R=4, source/target task similarity, and compact learned canonicalizing input map before the shared trunk. Freeze a new distinct fresh set before running. Against any proposed Mirror win, ordinary 4D linear/diagonal codes and independent LoRA remain compulsory controls.

## Source/replay

- [MAT01 frozen protocol](MAT01_FROZEN_PROTOCOL.json), [MAT02 independent frozen amendment](MAT02_FROZEN_PROTOCOL.json)
- [MAT01 source](source/run_mat01.py), [MAT01 unit tests](source/test_mat01.py), [MAT02 source](source/run_mat02.py), [MAT02 unit tests](source/test_mat02.py)
- [Complete per-world raw CSVs](results/) and [compact core](results/RESULTS_CORE.csv), [verification](results/VERIFICATION.json).
- All changes are restricted to isolated research branch. Main and worker-ready MA-255 queue remain untouched. No automatic status/claim modification.
'''
(base/'REPORT.md').write_text(text)
print(json.dumps({'MAT01':checks['mat01'],'MAT02':checks['mat02'],'report':str(base/'REPORT.md')},indent=2)[:1700])
