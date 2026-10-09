# MA-545 — Function-vector activation MoE without weight experts

Status: SCREENING; frozen protocol, no development metrics yet.

## H — Falsifiable hypothesis

A query-context router can select a support-extracted function vector from a shared 16-task bank and retain most oracle-vector quality on held-out relation queries, while storing far less than 16 full layer MLP experts. The critical native control is a routed per-task output-bias expert: adding a vector at the residual output is exactly a bias delta at that layer. Any equality with that control rules out Mirror-specific attribution even if activation interventions improve task behavior.

## Prior art and delta

PA99 reports function identity in attention-head activation vectors and partial composition; PA100 is a simple contrastive activation-addition baseline. MA-516 established support-derived Pythia task vectors on the same relation task bank, but rank-4 compression was quality-limited and exactly reproduced by ordinary PCA. MA-539/540 show that support codes need a stronger native comparison. MA-545 changes the question from compressing one task vector to routing among many activation-space behavior vectors, and explicitly compares a same-router weight-bias expert with identical outputs.

## T — Frozen task and controls

Pinned EleutherAI/pythia-70m-deduped revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, hidden dimension 512, intervention after block 3 (hidden-state output index 4). All 16 fixed relation directions are used. Per relation, eight examples form the FV extraction support, four disjoint examples calibrate a content router, and four disjoint examples are audit queries. Router centroids are average normalized hidden states at the final `Output` delimiter; inference receives the prompt, not its task ID. The top-2 router is fixed to temperature .05.

Controls: no intervention; oracle task-ID FV; query-context top-1 and top-2 routed FVs; oracle per-task output-bias experts; same-router output-bias experts; 16 full layer-3 MLP copies as a bytes-only structural reference. The output-bias control selects the exact same vector at the same layer, so functional output equality is expected and tests native attribution.

## Mirror insertion and paid state

- Physical object: one frozen Pythia-70M base.
- View address: one 512-d support-derived vector per task.
- Router address: one 512-d context prototype per task.
- Activation bank cost: all 16 vectors, router prototypes, IDs and metadata; actual total deployment also includes the pinned model and tokenizer.
- Weight control: stores the same vector bank as per-task residual-output bias deltas, with the same router state.
- Full per-task MLP copies are stored and byte-counted only; no quality is attributed to those untrained copies.
- Support examples are charged as extraction inputs/compute, but not persisted in the inference payload after vectors and router prototypes are compiled.

## Development/fresh boundary

Development seeds: 54501, 54502. Fresh seeds: 54511–54513. One fixed method configuration, no alpha/layer/router tuning. Open fresh only if in both dev seeds oracle FVs beat no-intervention by >=.05 top-1 accuracy or >=.15 gold log-probability nats and top-1 routing is within .05 accuracy/.15 nats of the oracle. Fresh task facts remain the same; split seeds make support, calibration and audit pairs disjoint.

Mirror attribution requires routed FV to beat the same-router output-bias expert by >=.02 accuracy or >=.10 gold log-probability nats at comparable bytes. An exact output equality is recorded as an attribution FAIL.

## C / U

Strong counter-hypothesis: the vectors are ordinary residual biases, and routing is just task classification; the output-bias control reproduces the full function with identical state. Unconfirmed: open-ended generation, broader task families, longer contexts, modern larger models, learned/soft routing, and whether a shared nonlinear intervention basis beats native low-rank weight adapters.
