# MTPM-001: Mirror multi-token drafts and direction mixing — audited CPU research

Date: 2026-10-17 JST. **Status: MECHANISM / SYNTHETIC ONLY.** Prior MA-249 and TM001 statuses unchanged; not an adopted Mirror architecture, not a natural-language LM, and **no end-to-end speculative decoding speedup claim**.

## 1. Bottom line

- **Positive mechanism:** Hidden-branch four-token drafts learned with two latent plans. The Mirror-plan variant reached an oracle-target p/q accepted-prefix mean of **3.795/4** with an actual serialized inference payload of **46,192 B**, compared with **2.343/4** and **60,216 B** for separately parameterized joint-MTP heads. This is 23.3% fewer full-model bytes. It beat independent joint heads on hidden-branch joint NLL in all 3 fresh worlds at 900 updates.
- **Crucial native control:** simple shared gain also reached **3.641/4** accepted tokens at **46,702 B**. Mirror beat gain on joint NLL in only 2/3 worlds. **No universal Mirror-specific Pareto dominance.**
- **Direction mixing:** additional same-plan across-horizon self-attention failed its registered incremental-value gate **0/3 hidden worlds** versus plain Mirror and gain+attention jointly. A more complicated mechanism did not reliably improve quality.
- **Joint structure is valuable independently of Mirror:** two-plan native joint-MTP beat one-plan marginal MTP on hidden-branch sequence NLL in 3/3 worlds.
- **Duration boundary:** a new, *separately preregistered after primary results* long-budget follow-up (3 new worlds × 2 paired seeds × 4 methods = 24 models; 2,400 updates) again found Mirror lower NLL than independent joint heads in 3/3 worlds. It did NOT establish superiority over gain (2/3 worlds) or consistent value for direction attention.

## 2. Fixed design and isolation of Mirror parameter m

The physical object is one **three-prefix-token causal SDPA Transformer** (trainable embeddings, shared 32-wide QKV and output projection, GELU FFN and normalization). The observed tokens are current state, rule and branch indicator (revealed) or MASK (hidden). No true future tokens ever enter the forward pass. There are 16 states, four independently sampled transition rules, two trajectory branches, and **P=4 future positions**. Labels are generated from a rule-and-branch transition held constant for the whole packet. Each future location has one shared-output low-description Mirror rotation code; hidden mode uses two candidate-plan × four-horizon code slots. Mirror codes are **a full 2×4 angle table, not yet a factorized branch×position generator**.

Compared: tied head, full independent (plan×horizon) head, shared diagonal gain, Givens Mirror, each plus separate same-plan horizon-latent SDPA (gain_attn/mirror_attn), shared rank-2 residual, plus a one-plan marginal MTP in hidden conditions. The direction mixer shares information **only across future-position slots within the same candidate plan**; this is not full all-Mirror/all-branch attention. All models train their whole shared trunk end-to-end from the same initial trunk tensors and the same mini-batch index stream for paired seed/world comparisons. Variable head-initialization effects remain.

Conditions: cyclic revealed, random-permutation revealed, random-permutation hidden. Dev: world 98201, 12 fits at 350 updates; fixed common AdamW LR 0.01, weight_decay 1e-4. Primary fresh: **worlds 98301, 98302, 98303**, init seeds 27101,27102, 132 models (7 methods in each revealed condition, 8 hidden), 900 updates, batch64, 2,048 train / 512 validation / 1,024 IID test examples each. IID examples may repeat finite state-rule combinations: this is **not** held-out rule generalization. Evaluation checkpoints fixed at 150/450/900 updates. Distinct follow-up long worlds **98401..98403**, same paired seeds, methods independent/gain/mirror/mirror_attn, 2,400 updates, checkpoints 900/1600/2400. Primary and follow-up source/protocol locks were saved before opening their respective evaluation worlds; no fresh-set hyperparameter tuning.

### Probability contract

For a candidate plan c and future position k, the draft model outputs a categorical q_c,k(.|observed prefix). A P-token draft proposal has joint probability

\[q(y_{1:P}|x)=\frac1C\sum_{c=1}^C\prod_{k=1}^P q_{c,k}(y_k|x).\]

Train by minimizing the full joint negative log likelihood divided by P. C=1 for revealed, C=2 for hidden (except marginal MTP C=1). Hidden branch identity is **never provided as an input**; the two plans are trained as a symmetric latent mixture. Native independent 2-plan heads are a particularly important control because they have enough per-slot parameters but different training dynamics.

**Oracle-target p/q accepted-prefix diagnostic:** draw a four-token candidate sequentially from the exact conditional distributions of q, updating the latent-plan posterior after each proposed token. For each draft token, compare the known synthetic target transition distribution p to q and accept with probability min(1,p(token)/q(token)). Stop on first rejection. Test on 1,024 IID contexts × 6 proposal replicates for every model, with a fixed RNG seed per paired world/initialization. The target p is the correct autoregressive conditional mixture over hidden branches, updated after every accepted token. This measures genuine one-block p/q acceptance probabilities against an **oracle synthetic target**, but no residual-distribution correction token is sampled and no actual Transformer verifier forward, bonus token, cache rollback, or decoder throughput is timed. The greedy ground-truth prefix metric is recorded separately; do NOT substitute it for p/q acceptance.

## 3. Primary data: three fresh worlds, 2 initializations each

Each row is the median across three independent world means of the two paired initializations. Lower joint NLL is better. Real inference bytes include the learned trunk, all codes/heads, indices/config serialization. Models are NOT compared at matched optimization wall time.

### Random permutations, hidden branch — primary key test

|方式|4-token joint NLL|受理数／4|受理4トークン率|実保存B|
|---|---:|---:|---:|---:|
|marginal_mtp|4.0386|1.4359|0.1315|51,768|
|independent|2.4959|2.3433|0.4980|60,216|
|gain|1.1038|3.6413|0.8861|46,702|
|mirror|0.9121|3.7953|0.9325|46,192|
|gain_attn|1.2536|3.5456|0.8640|47,024|
|mirror_attn|1.0205|3.7043|0.9039|46,514|
|lowrank|2.4659|2.4511|0.5196|49,006|
|tied|7.6582|0.4061|0.0082|45,425|

### Random permutations, branch revealed

|方式|4-token joint NLL|受理数／4|受理4トークン率|実保存B|
|---|---:|---:|---:|---:|
|independent|0.1246|3.8845|0.9648|51,768|
|gain|0.1203|3.8752|0.9596|46,190|
|mirror|0.1635|3.9150|0.9744|45,936|
|gain_attn|0.2201|3.8301|0.9482|46,512|
|mirror_attn|0.1149|3.8895|0.9660|46,258|
|lowrank|0.2896|3.8139|0.9438|47,470|
|tied|5.4137|0.5269|0.0563|45,425|

### Structured cyclic transitions, branch revealed

|方式|4-token joint NLL|受理数／4|受理4トークン率|実保存B|
|---|---:|---:|---:|---:|
|independent|0.1606|3.8244|0.9455|51,768|
|gain|0.2541|3.7556|0.9239|46,190|
|mirror|0.1329|3.8633|0.9568|45,936|
|gain_attn|0.0602|3.9390|0.9805|46,512|
|mirror_attn|0.0629|3.9428|0.9824|46,258|
|lowrank|0.1956|3.8431|0.9505|47,470|
|tied|5.6499|0.3923|0.0098|45,425|

Pre-registered gates: Mirror vs independent quality/byte on revealed cyclic **FAIL 1/3** worlds, on revealed random **PASS 2/3** worlds. Mirror cross-direction-attention advantage over plain Mirror and gain-attn jointly **FAIL 0/3** worlds on hidden random. Native two-plan joint model over independent one-plan marginal model **PASS 3/3** worlds for hidden sequence likelihood, which is not a Mirror-specific finding. The revealed-random PASS does not imply Mirror beats the gain native baseline or a fixed-byte compute frontier.

## 4. Longer-budget follow-up, separately locked, not retroactive tuning

|world|independent NLL|gain NLL|mirror NLL|mirror_attn NLL|mirror p/q受理数|
|---|---:|---:|---:|---:|---:|
|98401|2.4911|1.1428|1.0810|0.8726|3.6482|
|98402|2.0321|0.7262|0.8111|0.8139|3.8984|
|98403|1.9082|1.4354|1.2250|3.2201|3.5249|

This part was chosen after examining the primary result, and opened new independent teacher worlds. Mirror beats full independent two-plan heads in **3/3 worlds** at 2,400 fixed updates. The stronger *simple gain* beats Mirror in one world. The direction-attention version is catastrophically unstable for one world/seed (final joint NLL >5) and thus cannot claim reliable superiority. All methods use 2,400 updates, 153,600 example presentations. No clock/FLOP matched training conclusion.

## 5. Storage, inference time, uncertainty

Actual full serialized bytes in hidden mode: independent 60,216; marginal independent 51,768; tied 45,425; gain 46,702; Mirror 46,192; gain+attention 47,024; Mirror+attention 46,514; rank-2 low-rank 49,006. No memory allocator/optimizer bytes are part of inference archives; training checkpoint bytes are separate.

Single-thread CPU batch1 unfused torch-forward snapshot for hidden world98301/init27101: independent 0.2368 ms; gain 0.2660 ms; Mirror 0.3228 ms; Mirror+attention 0.3323 ms. The Mirror draft **was not faster** than independent on this one-forward microbenchmark. Real end-to-end speculative decoding can shift the balance because verification and batching dominate; not tested.

Uncertainty is evaluated at the **three independent world level**, not six independent worlds. The primary hidden (Mirror minus independent) paired world mean joint NLL differences are [-0.7736, -1.5838, -1.6034]; mean difference -1.3203, standard uncertainty (sample SD / sqrt(3)) 0.2734, exploratory 95% Student-t interval [-2.4966,-0.1439], using df=2 and k=4.303. This interval does not cover distribution shift to real language, hardware, model scale, or other synthetic teachers. NLL is dimensionless natural logarithm units ('nats'); serialized bytes are bytes; times are milliseconds/seconds; branch probabilities and acceptance rates are dimensionless. Numeric additions and likelihoods are dimensionally consistent.

All seed-specific and world-specific results, NLL-learning curves, acceptance statistics and runtime groups are in CSVs and the full evidence archive; medians alone do not show the full variability.

## 6. Verification, scope, and counter-hypotheses

- **14 unit tests PASS**, covering Mirror norm invariance, hidden-token non-leak, joint-probability normalization, equal trunk initialization, deterministic training, serialized payload restore, exact oracle acceptance counterexamples and hidden two-plan test.
- All **132 primary and 24 long** inference payloads verified byte length and SHA-256, reloaded state tensor SHA-256, finite inference logits, and recomputed all recorded test metrics. Maximum absolute logged metric difference: **0** in this same-host replay.
- At least one primary and one long model were re-trained from scratch, matching all stored metric checkpoints, tensor SHA-256 and serialized inference payload SHA-256 exactly. This is **not** an independent external replication.
- Partial model-parallel orchestration experienced tool timeout in one 900-update job and two long-budget batches; missing fits were rerun with the SAME frozen source, protocol and seeds. No fresh hyperparameter changes. Per-fit evidence records are intact.
- No genuine natural language, no pretrained causal LM, no real verifier rollout, no cache/fused GPU test; IID overlap by finite state-rule support; Givens is only a small per-slot transform; the native gain is competitive. No Mirror-specific universal speed/quality/byte advantage is established. Follow-up remains a new, separate prospective intervention.

**Strongest counter-hypothesis:** constraining all horizon heads to a common geometric/diagonal basis regularizes a finite dataset, allowing joint packet modes to learn faster under limited updates. A native gain attains similar results; the benefit could be due to compact parameter sharing rather than Mirror geometry. More diverse datasets and matched FLOPs/seconds are required before promotion.

## 7. Reproduce and next experiment

For a clean checkout of the source package, run from its root:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests
python run.py --phase dev --workers 3
python run.py --phase fresh --workers 4
python audit.py
python long_run.py --world 98401 --workers 2
python long_run.py --world 98402 --workers 2
python long_run.py --world 98403 --workers 2
python long_audit.py
```

The full evidence package includes all actual model payloads, per-run checkpoints and provenance logs. If the source package does not include raw output files, use new output directories to avoid overwriting the audit evidence. On the 2,400-update follow-up, independent per-model execution was used to recover from orchestration timeouts.

A new experiment should use an actual pretrained tiny causal LM as target verifier, compare against well-tuned Medusa/Hydra/EAGLE-compatible native heads, use accurate proposal acceptance and correction/bonus-token accounting, measure real KV cache and end-to-end latency, and probe whether horizon-code factorization and cross-plan attention help. Keep token-axis causal masking distinct from future-slot latent attention. This study's positive result is limited to a synthetic conditional process.

**ERROR CHECK:** Distinguish actual p/q acceptance from greedy truth agreement, equal updates from equal wall/FLOPs, target-known oracle from deployable language verifier, joint mixture learning from per-slot independent predictions, and compression in bytes from actual speed. Do not overwrite MA-249/TM001 or earlier failure records.
