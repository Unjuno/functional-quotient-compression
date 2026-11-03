# MA-545 — Function-vector activation MoE without weight experts

Status: **FAIL** for Mirror-specific attribution (frozen development and three fresh seeds completed).

## H — Falsifiable hypothesis

A query-context router can select a support-extracted function vector from a shared 16-task bank and retain most oracle-vector quality on held-out relation queries, while storing far less than 16 full layer MLP experts. The critical native control is a routed per-task output-bias expert: adding a vector at the residual output is exactly a bias delta at that layer. Any equality with that control rules out Mirror-specific attribution even if activation interventions improve task behavior.

## Prior art and delta

PA99 reports function identity in attention-head activation vectors and partial composition; PA100 is a simple contrastive activation-addition baseline. MA-516 established support-derived Pythia task vectors on the same relation task bank, but rank-4 compression was quality-limited and exactly reproduced by ordinary PCA. MA-539/540 show that support codes need a stronger native comparison. MA-545 changes the question from compressing one task vector to routing among many activation-space behavior vectors, and explicitly compares a same-router weight-bias expert with identical outputs.

## T — Frozen task and controls

Pinned EleutherAI/pythia-70m-deduped revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, hidden dimension 512, intervention after block 3 (hidden-state output index 4). All 16 fixed relation directions are used. Per relation, eight examples form the FV extraction support, four disjoint examples calibrate a content router, and four disjoint examples are audit queries. Router centroids are average normalized hidden states at the final `Output` delimiter; inference receives the prompt, not its task ID. The top-2 router is fixed to temperature .05.

Controls: no intervention; oracle task-ID FV; query-context top-1 and top-2 routed FVs; oracle per-task output-bias experts; same-router output-bias experts; 16 full layer-3 MLP copies as a bytes-only structural reference. The bias control gates a per-task MLP output-projection bias delta at the query position. It is algebraically identical to the residual intervention; maximum absolute candidate-score difference <=.01 allows for floating-point addition order.

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

## Results and decision

### H / T / D / C / U

- **H:** Test whether 16 support-derived activation views plus a query-context router provide useful held-out relation behavior without 16 copies of layer-3 MLP weights, and whether that gain is specific to Mirror activation views rather than a native output-bias expert.
- **T:** Frozen Pythia-70M, 16 relation tasks, 8 support + 4 router-calibration + 4 audit pairs per task, 2 development seeds and 3 sealed fresh split seeds. Compared no intervention, shared mean FV, oracle/task-routed top-1 and top-2 FVs, and algebraically matched oracle/same-router output-bias controls. The 16-copy MLP comparison is bytes-only, not a quality result.
- **D:** **FAIL for Mirror-specific attribution.** Top-1 routing identified the task in 100% of calibration/audit queries and exactly matched the oracle FV's measured task-ranking outputs. But the native bias control matched the activation FV within the frozen 0.01 numerical tolerance on all five seeds. Top-1 accuracy fell relative to no intervention in all five seeds, while mean gold log-probability improved by 3.39–3.65 nats on every fresh seed. Top-2 and shared-mean variants did not reverse the attribution result.
- **C:** The vector is simply a position-gated residual/output bias, and the router is a task classifier; the activation-space parameterization adds no distinct function beyond the native bias control. Improved gold likelihood alongside lower exact-choice accuracy also shows that likelihood gain does not imply better decisions here.
- **U:** This is a small fixed relation-ranking screen. It does not establish open-ended generation, broader task transfer, larger-model effects, or learned routing. Per-method latency is not isolated; only complete seven-method evaluation wall time and extraction wall time are reported.

### Facts

- Across five seeds, query-context top-1 router task accuracy and top-2 task inclusion were both 1.0. Routed top-1 FV metrics exactly equal oracle FV metrics.
- Fresh mean held-out accuracy: no intervention **0.4063**; routed/oracle FV and matched bias **0.3750**; top-2 FV **0.3802**; shared mean FV **0.4063**. Accuracy fell by 1.6–4.7 percentage points for routed top-1 FVs on every fresh seed.
- Fresh mean gold log-probability: no intervention **−10.4791**; routed/oracle FV and matched bias **−6.9884**; top-2 FV **−7.0791**; shared mean FV **−8.6904**. Routed top-1 FV improved gold log-probability by 3.43–3.65 nats per seed.
- On each fresh seed FV vs same-router bias maximum candidate-score difference was 0.00143–0.00301, within the preregistered 0.01 tolerance. Their serialized state differs by 4 bytes (68,396 B FV, 68,400 B bias); the bias path is algebraically equivalent and does not need more compute.
- Fresh FV payload is 68,396 B (full deployment 168,213,020 B including the common model/tokenizer); 16 full layer-3 MLP copies are 134,400,454 B incremental bytes, but are only a storage reference without trained quality. Support extraction used 256 FV forward calls and 64 router-calibration forward calls per seed. Total seven-method audit evaluation took 49.47–50.94 s; support extraction took 16.31–20.12 s on the recorded CPU setup. Each seed evaluated 64 audit prompts and made 448 model calls across methods. Each scored method costs 16 cosine dot products of width 512 for query routing plus a full candidate-batch model pass; top-2 also adds a second vector and softmax mixing. Per-method wall time is not isolated.

### Interpretation

Support-derived function vectors are useful for improving the assigned gold answer's likelihood on this fixture, with a compact stored intervention bank. This is an activation-intervention utility result, not a Mirror-specific gain: the direct native bias control reproduces it. The bank is substantially smaller than the bytes-only 16-MLP reference, but no quality comparison against trained independent MLP experts was made. Accuracy loss means the intervention cannot be described as an unqualified quality improvement.

### Hypothesis for follow-up

A nonlinear, input-dependent shared activation basis may escape the output-bias alias, but must beat matched native low-rank/adapter or gated-weight controls on actual payload bytes, compute and held-out task quality before being called Mirror-specific.
