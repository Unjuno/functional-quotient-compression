# Independent Mirror research runbook (worker queue untouched)

> This is an isolated laboratory branch, **not a request for the active worker to scan or claim these rows**. The canonical worker branch `research/mirror-application-worker-ready-20261007` remains authoritative. New entries MA-1156..1164 exist ONLY here until an explicit safe promotion / ID collision recheck.

## 1. Minimal reproduction with no active-worker interference

Use a separate disposable checkout or Git worktree; **never** change the active worker's current directory, git HEAD, queue or lock.

```bash
git fetch origin research/mirror-isolated-protocols-rebased-20261008
git worktree add /tmp/mirror-isolated-20261008 origin/research/mirror-isolated-protocols-rebased-20261008
cd /tmp/mirror-isolated-20261008
python experiments/mirror_applications/check_registry_integrity.py
python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/selfcheck.py
python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/algebra_smoke.py
```

The worktree commands are OPTIONAL if an isolated checkout already exists; these do not start the current worker or use GPU. Repository integritiy must be checked **on this branch**; its registry has 1,164 entries, PA01..424, with existing completed claims unchanged.

## 2. Existing two CPU pilots (evidence, not MA completion)

```bash
ROOT=experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008

# Exact native-GVA cache alias, unsafe changed-prefix counterexample, and factorized-code parity
python "$ROOT/pilots/gva_alias_stage0/source/run_stage0.py" --phase dev --output /tmp/gva-pilot-dev
python "$ROOT/pilots/gva_alias_stage0/source/run_stage0.py" --phase fresh --output /tmp/gva-pilot-fresh

# Gauge tangent nullspace dimension with and without 2D RoPE
python "$ROOT/pilots/gauge_rank_stage0/source/run_gauge.py" --phase dev --out /tmp/gauge-pilot-dev
python "$ROOT/pilots/gauge_rank_stage0/source/run_gauge.py" --phase fresh --out /tmp/gauge-pilot-fresh

python "$ROOT/pilots/gva_alias_stage0/source/test_stage0.py"
python "$ROOT/pilots/gauge_rank_stage0/source/test_gauge.py"
```

**Important:** The two tests that read `fresh_raw.csv` expect those CSVs in the **same directories as the scripts**, while the read-only reproduction commands above deliberately send new outputs to /tmp to avoid modifying Git-tracked audited files. To verify without modifying tracked audit data, copy each test script and matching `source/*.py` plus its `RESULTS_FRESH.csv` renamed to `fresh_raw.csv` into a disposable scratch directory. Or run the test scripts in an isolated worktree in which fresh CSV snapshots have been copied alongside them. Never overwrite the original audited CSV in Git. CPU NumPy + PyTorch are required for these two pilots; no GPU is required. Exact numbers (other than timing) should reproduce within tolerance if the dependency versions and fixture are the same.

Stage-0 results already on this branch:
- [GVA result](pilots/gva_alias_stage0/REPORT.md): algebra and zero-copy alias PASS; native single-cache GVA already provides the same alias; identical native factorized-code bytes and slower eager CPU Mirror path ⇒ **no incremental Mirror value in this synthetic scope**.
- [Gauge result](pilots/gauge_rank_stage0/REPORT.md): bare Q/K Jacobian rank12/nullity4; nontrivial 2D RoPE joint rank14/nullity2; exact gauge code is not a new function.

Do NOT mark MA-1156 or MA-1162 complete; their full natural-language, native-paper, byte-near comparison remains untested.

## 3. Exactly-one-experiment route for future execution

Each row is a standalone readme plus frozen `PROTOCOL.json` and `STATUS.md` under `plans/`. For any selected ID:

1. Read only that ID's `plans/MA-*/README.md` and its corresponding JSON; no traversal of all literature notes is required for the first synthetic mechanism test.
2. Confirm this staging branch's ID is not newly allocated by the concurrent worker; on a collision, rename this staging hypothesis before any merging (never overwrite the worker row).
3. Allocate a separate scratch experiment directory, with `source/`, `tests/`, `RESULTS_CORE.csv`, and `VERIFICATION.json`; freeze before fresh data.
4. Faithfully reproduce the **native** prior-art comparator or label it UNREPRODUCED; run shared-only, matched native low-rank/gate/linear-basis, Mirror, and independent upper reference.
5. Select hyperparameters from seeds `11,12,13` on support/dev only, then execute each fresh seed `101..105` once without any tuning.
6. Report quality in task-native units, actual serialized inference bytes for all paid objects, persistent cache alias memory where relevant, training compute and P50/P95 latency on explicitly identified hardware/dtype/batch. A small positive mechanism effect is not a capacity or Pareto proof.
7. Record scoped negative findings even when the main Mirror hypothesis fails; never discard an adversarial counterexample.

| ID | Read exactly this folder first | Earliest justified measurement |
|---|---|---|
| MA-1156 | `plans/MA-1156/` | gauge-aware functional Jacobian and held-out code loss |
| MA-1157 | `plans/MA-1157/` | path-conditioned code drift versus Procrustes |
| MA-1158 | `plans/MA-1158/` | fingerprint collisions and false equivalence |
| MA-1159 | `plans/MA-1159/` | permutation-equivariant unseen-task code quality |
| MA-1160 | `plans/MA-1160/` | held-out ordered pair loss and commute control |
| MA-1161 | `plans/MA-1161/` | native GVA versus head-map code bytes and NLL |
| MA-1162 | `plans/MA-1162/` | physical cache alias, target re-prefill, role-switch P95 |
| MA-1163 | `plans/MA-1163/` | cross-architecture task/model holdout utility |
| MA-1164 | `plans/MA-1164/` | LRKV native residual versus m-code byte and NLL frontier |

## 4. What constitutes a result

- **H:** measured outcome and predeclared margin under fixed data/hardware.
- **T:** exact support/dev/fresh identity split, (n_{\min}=5) initial worlds, physical storage and compute accounting.
- **D:** PASS/FAIL/UNCERTAIN assessed jointly versus the **nearest native** alternative.
- **C:** counterhypothesis must explicitly include "native sharing or low-rank is sufficient."
- **U:** paired seed/task variation, numerical conditioning, actual serialized byte count, CPU/GPU clock/dtype. Correlated uncertainties require covariance; the k=2 multiplier on 5 seeds is indicative only.

A claim becomes ADOPTED only with separate stronger replication and a useful Pareto improvement. Source PDFs are useful for later scientific publication, but the independent README/PROTOCOL captures enough to implement the first falsification harness.

## 5. Source/reproducibility contract

Prefer container CPU for algebra, SVD, routing and synthetic tasks. Escalate to GPU ONLY for actual trained tiny LM/native architecture and decode kernels; report unavailable hardware as BLOCKED, not as performance gains. Do not edit vendor `third_party/nanoGPT`, `WORKER_QUEUE.md`, `CONTEXT_ROUTER.md`, `WORKER_START_HERE.md`, the canonical worker branch, or `main`.

Previously collided MA-1116..1123/PA382..390 in the old independent branch are not valid references for these hypotheses. This staging branch was reconstructed from canonical SHA `c935a903daca5c7d1d48aa50d05b5bd50f239cba` with clean IDs MA-1156..1164 and PA414..424.
