# Worker Start Here — Mirror Application Program

This repository contains several historical research lanes. Do not infer the current task from the oldest or largest directory.

## 1. Read in this order

1. `docs/phase2/CURRENT_STATE_2026-10-07.md`
2. `docs/phase2/MIRROR_APPLICATION_DESIGN_SPACE.md`
3. `docs/phase2/LATEST_WORKER_FINDINGS.md`
4. `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`
5. `experiments/mirror_applications/IDEA_REGISTRY.csv`
6. `experiments/mirror_applications/FIRST_QUEUE.md`
7. `experiments/mirror_applications/EXPERIMENT_CONTRACT.md`
8. `roadmap/MIRROR_APPLICATION_ROADMAP.md`

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
