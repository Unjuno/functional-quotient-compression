# Worker Start Here — Mirror Application Program

This repository contains several historical research lanes. Do not infer the current task from the oldest or largest directory.

## 1. Read in this order

1. `docs/phase2/CURRENT_STATE_2026-10-07.md`
2. `docs/phase2/MIRROR_APPLICATION_DESIGN_SPACE.md`
3. `docs/phase2/LATEST_WORKER_FINDINGS.md`
4. `experiments/mirror_applications/CONTEXT_ROUTER.md`
5. `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`
6. `experiments/mirror_applications/IDEA_REGISTRY.csv`
7. `experiments/mirror_applications/FIRST_QUEUE.md`
8. `experiments/mirror_applications/EXPERIMENT_CONTRACT.md`
9. `roadmap/MIRROR_APPLICATION_ROADMAP.md`

Read historical reports only when the selected MA experiment points to them.

## 2. Select exactly one MA ID

Do not start with a family name such as "Mirror-MoE". Pick one stable ID, e.g. `MA-003`.

Create:

`experiments/mirror_applications/<id-lowercase>_<short_name>/`

Never reuse an ID for a different hypothesis.

## 3. Before coding

Create these files first:

- `README.md` — hypothesis, prior-art delta, controls, success/failure gates;
- `PROTOCOL.json` — frozen train/dev/fresh split, storage and compute contract;
- `STATUS.md` — current stage and last verified commit.

The experiment README must answer:

- What physical object is being shared?
- What does the Mirror/View coordinate change?
- What logical objects are claimed?
- What is the cheapest simpler control?
- What prior-art item is closest?
- What result would falsify the hypothesis?

## 4. Evidence lanes

Keep these separate:

- **MECHANISM** — controlled synthetic task;
- **LANGUAGE** — nanoGPT/tiny-LM external validity;
- **STORAGE** — actual serialized bytes;
- **RUNTIME** — measured active compute / wall time;
- **CAPACITY** — near-convergence quality at fixed bytes; requires stronger evidence than a fixed-update win.

Never promote a result from one lane into another without a new experiment.

## 5. Common baselines

Use `third_party/nanoGPT/` as the stable causal-LM baseline when appropriate. Do not edit the vendor copy for an experiment. Implement variants in the experiment directory or a thin project module.

Default control hierarchy:
1. dense/shared baseline;
2. existing non-Mirror method being replaced;
3. byte-near low-rank/gate/shared-basis control;
4. unrestricted independent-object upper control when practical.

## 6. Storage and compute

Storage claims require actual serialized inference payload bytes.

Count:
- learned tensors;
- Mirror codes;
- routers;
- bases;
- indices;
- codebooks;
- reconstruction metadata.

Report separately:
- tokens/examples seen;
- optimizer updates;
- active MAC/FLOP proxy;
- isolated wall-clock;
- inference throughput if relevant.

Fixed-update superiority is learning-efficiency evidence, not automatically capacity.

## 7. Development discipline

- Development data/worlds choose hyperparameters.
- Fresh/audit data never choose hyperparameters.
- Preserve negative results.
- Do not rewrite historical reports to make the current idea look better.
- If a simpler control matches the candidate, record the candidate as non-Mirror-specific or FAIL as appropriate.
- If a result depends on an executor, oracle address, private mask, intermediate target, or extra forward pass, state it prominently.

## 8. Experiment completion

Every finished experiment should contain:

```text
README.md
PROTOCOL.json
STATUS.md
RESULTS_CORE.csv
VERIFICATION.json
source/
tests/
```

Large checkpoints may remain outside normal Git history, but record hashes and exact reconstruction requirements.

Update the corresponding row in `IDEA_REGISTRY.csv` only after the report and verification files exist.

## 9. Branching

Use a dedicated research branch:
`research/<ma-id>-<short-name>-YYYYMMDD`

Do not merge to `main` as part of an experiment unless explicitly requested.

## 10. Scientific language

Allowed:
- "wins 3/3 fresh worlds at fixed updates";
- "uses 18% fewer serialized bytes";
- "promising";
- "failed the gate".

Not allowed without evidence:
- "capacity multiplier";
- "equivalent to N independent experts";
- "free compute";
- "general LLM compression";
- "proves natural-language rule reuse".


## Concurrent registry edits

Before adding a new MA candidate, re-read `IDEA_REGISTRY.csv` from the current branch and allocate IDs starting at **max existing MA ID + 1**. Never reserve an ID from memory or an older checkout. After writing, re-read the registry and verify zero duplicate IDs. If concurrent additions collide, preserve both hypotheses and renumber the later addition rather than deleting either one.


## 11. Candidate-specific context

### KV-cache candidates

For MA-691..700 and any later cache-transform candidate, read:
`docs/phase2/MIRROR_KV_CACHE_REUSE.md`.

Before coding, classify the hypothesis as one of:
1. identical-cache placement;
2. exact analytical cache transform;
3. canonical latent/cache with View-specific readout;
4. approximate learned cache translation.

Do not mix these four evidence classes. Measure physical cache aliasing/storage separately from prefill/switch latency and attention compute.

### MA-701..770 cross-domain expansion

Before implementing any MA-701..770 candidate:
1. read the exact registry row and all referenced PA entries;
2. read the "Eighth literature sweep" section in `docs/phase2/MIRROR_APPLICATION_RESEARCH_NOTES_2026-10-07.md`;
3. use the native domain metric/harness when the candidate is not a language-model mechanism — do not force FNO/DeepONet, GNN, diffusion or NCA hypotheses through nanoGPT merely for uniformity;
4. preserve the common evidence contract: actual serialized bytes, active compute, wall-clock calibration, strongest simple control, and fresh/audit split;
5. for factorized coordinates, hold out combinations rather than only evaluating seen IDs;
6. for Bayesian/ensemble candidates, report calibration and member diversity;
7. for dynamical/NCA candidates, report stability and rollout/recovery failure modes;
8. treat MA-770 as a cross-domain benchmark only after representative component mechanisms have been screened.

### MA-771..825 invertible/reuse/dynamics expansion

Before implementing any MA-771..825 candidate:
1. read every referenced PA item and identify whether the candidate changes physical storage, logical multiplicity, routing, state rank or only coordinates;
2. for invertible Views, report forward quality, inverse/cycle numerical error, transform latency and any convergence/Lipschitz constraint;
3. for MoE reuse/upcycling, report physical expert count, logical expert count, router bytes, load balance, active K, continued-pretraining compute and expert similarity/diversity;
4. when using merge/prune results as a physical basis, compare against merge-only/prune-only and small-from-start controls;
5. for low-rank recurrent/Koopman candidates, report latent/effective rank, stability spectrum where meaningful and long-horizon error;
6. for Vector-Network/dynamic-atom candidates, count per-input inference iterations and active atoms;
7. for robot/control candidates, report hardware, rollout horizon, inference latency, bytes per added skill and OOD composition;
8. for matrix-memory candidates, do not interpret failure below a proven/known rank threshold as evidence against optimization alone;
9. for programmable graphs, count topology/program metadata and dynamic execution cost as part of the Mirror state.

### MA-826..875 edge/tangent/adaptive-state expansion

Before implementing any MA-826..875 candidate:
1. KAN candidates must count stored basis/function parameters and actual function-evaluation runtime; coefficient count alone is insufficient.
2. NTK/tangent candidates must measure the relevant linearization or kernel approximation error before interpreting representational failure.
3. AI-engram candidates must retain causal specificity, reactivation, sufficiency and necessity checks; weight similarity is not enough.
4. online-state candidates must separate persistent bytes, writable state bytes, write FLOPs and reset/restore cost.
5. DeltaNet/fast-memory candidates must include long-context retrieval and state-drift tests.
6. learned-subspace/manifold candidates must count all endpoints, basis vectors and Bezier/control parameters as physical storage.
7. Mesh candidates must preserve the observation-only boundary and include communication latency/message bytes.
8. structural-composition candidates require held-out module/rule combinations and sample-efficiency measurements.
9. stable-synapse controls are mandatory when a claimed fast-weight benefit might be achievable by gain/context modulation alone.

### Recurrent/depth candidates

MA-247 showed that even an aligned Givens teacher can fail a fixed-budget recurrent optimization screen. Include a scalar/static-LoRA optimization control and do not infer representational impossibility from failed convergence.

## 12. Research support intake — use at the next safe boundary

[PR #27: computation reuse and correctness guards](https://github.com/Unjuno/functional-quotient-compression/pull/27) adds 16 follow-up subtests for existing MA-003/672/691..700, with proofs, 19 unit tests and a 200-check numerical replay. The package is on `research/mirror-compute-reuse-support-20261007`, under `experiments/mirror_applications/research_intake/compute_reuse_20261007/`. It is not a language-quality or runtime result.

For MA-003, inspect shared-projection fusion, sign-View gate/bypass equivalence and antipodal cancellation. For KV candidates, inspect common-map value fusion, original key-width temperature after latent absorption, missing-information counterexamples and source-token cache provenance.

Do not interrupt a frozen run or change audit seeds/gates to incorporate this intake. Check dedicated experiment branches before claiming work: at the inspected snapshots MA-248..251 were already reported complete on the MA-251 branch, while the shared board still listed MA-248 next. Preserve both histories; do not replace the expanded global registry with an older worker checkout. The PR uses local CR subtest IDs, not newly reserved global MA IDs.
