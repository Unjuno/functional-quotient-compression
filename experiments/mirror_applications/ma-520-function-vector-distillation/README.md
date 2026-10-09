# MA-520 — Function-vector distillation into a shared code decoder

Status: **FAIL** (frozen development screen; fresh seeds sealed). Branch: `research/ma-520-function-vector-distillation-20261009`.
Prior art: PA99, *Function Vectors in Large Language Models*. MA-516 found useful explicit FVs but rank-four PCA missed the gold-logprob gate and exactly aliased native PCA.

## H — Hypothesis

A learned rank-four shared decoder trained on FV teacher targets would preserve causal held-out effect within 0.10 mean gold-logprob nats and 0.05 accuracy of explicit FVs, use at most half the FP32 bank bytes, and beat native rank-four PCA by at least 0.10 gold-logprob nats in both seeds.

## T — Execution

We used pinned `EleutherAI/pythia-70m-deduped`, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, weights SHA-256 `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd`; the MA-516 extraction, task splits and evaluator; 16 relation functions; decoder-fit task IDs 0–11; four held-out tasks 12–15; and development seeds 52001/52002. The learned mean-plus-linear rank-four decoder used exactly 2,000 full-batch Adam updates. Held-out codes were least-squares projections of FV teacher targets. Controls were no intervention, explicit FP32 and int8 FVs, and centered native PCA rank four. Fresh seeds 52011–52013 remain unopened. Amendment 1 corrected a results-emitter bookkeeping error that had charged 2,000 updates to the explicit FP32 control; the initial output is preserved under `runs/pre_amendment_1_explicit_update_counter/`, and the same development seeds were rerun with outputs unchanged.

The common model/config/tokenizer base was 168,144,624 B. Actual uncompressed NPZ payloads were 34,454 B (FP32 FVs), 10,188 B (int8 FVs), and 12,692 B (both learned decoder and PCA); all metadata and scales are charged. Int8 scale quantization had held-out relative RMSE .0155/.0156 and nearly unchanged causal metrics (held-out gold-logprob differences +.001677/+ .000241 nats versus explicit, with the same top-1 accuracy). The learned decoder fit took 1.204/1.135 s and its registered training multiply proxy was 49,152,000 per seed. See `RESULTS_CORE.csv` for all bytes, support/query tokens, wall time, accuracy, log probability, margin, cosine and reconstruction metrics.

## D — Decision

**FAIL** under the frozen gates. Learned rank four was 0.607/0.859 nats worse than explicit FVs in mean held-out gold log probability; it used 12,692 B versus 34,454 B, but did not meet the <=0.10 nat preservation threshold. Top-1 accuracy matched explicit (0.2188/0.2188 and 0.1875/0.1875). Against native PCA at identical 12,692 B, the learned decoder changed gold log probability by only +0.000159/−0.000128 nats, far below the required +0.10 nats. Int8 FVs used fewer bytes (10,188 B) and were within 0.0017 nat of explicit, with identical top-1 accuracy. Fresh data stayed sealed.

## C — Strongest counter-hypothesis

The learned decoder is an ordinary low-rank shared basis in a different parameterization. It converges to essentially the same held-out function space as PCA, while int8 quantization preserves the FVs far better at lower storage. Its optimization cost buys no measured task-quality gain.

## U — Boundaries

This is a constrained candidate-ranking screen on four held-out tasks in one 70M model, not broad reasoning or open-ended generation. It measures FV storage compression after support-based teacher extraction; it does not reduce extraction tokens or establish general code distillation.

## Evidence classification

- **Facts:** both frozen seeds, actual serialized payloads, causal scores and reconstruction metrics are in RESULTS_CORE.csv. Four tests pass; deterministic replay and split checks are recorded in VERIFICATION.json. Fresh seeds were not accessed.
- **Interpretation:** the learned code reduces bytes relative to FP32 FVs but misses the causal preservation threshold, gives no meaningful gain over native PCA, and loses to int8 FVs on storage and quality.
- **Hypothesis:** a task-aware or nonlinear decoder could alter the causal tradeoff, but that redesign is outside this frozen experiment.
