# MA-516 — Pythia function-vector compression

Status: **FAIL** (rank-4 gold-logprob preservation gate and Mirror-specific attribution).
Branch: research/ma-516-function-vector-mirror-compression-20261008
Prior art: PA99, Function Vectors in Large Language Models.
Protocol freeze: 75a80cd0f9e218bf7233e1544de39f057dc96187; Amendments 1 and 2 are recorded separately.

## H — Hypothesis

On four held-out in-context relation functions in Pythia-70m-deduped, projecting support-extracted function vectors into a rank-4 shared basis will preserve held-out query accuracy within 5 percentage points and mean gold-candidate log-probability within .10 nats of explicit vectors, using <=.50x their actual vector payload bytes in both development worlds.

Mirror insertion: extract one residual-stream function vector per relation task from support demonstrations, fit a centered rank-r basis on task IDs 0–11, and represent all functions (including IDs 12–15) by small codes. Model weights remain frozen.

## T — Executed protocol

Pinned model: EleutherAI/pythia-70m-deduped, revision e93a9faa9c77e5d09219f6c868bfc7a1bd65593c, weight SHA-256 3da388…a33fd. The common model+config+tokenizer payload is 168,144,624 B. Hidden dimension 512, six layers; intervention at hidden_states[4]. Each of 16 tasks uses eight support pairs and eight disjoint queries. Function vectors are leave-one-support-pair-out demonstration-minus-query-only activations. Tasks 0–11 fit the centered basis; task IDs 12–15 are held out from basis fitting. There are 32 held-out queries and 256 candidate-scoring sequences per seed.

Two amendments preserved the original runs: Amendment 1 corrected rank-1 method dispatch and added a regression test; Amendment 2 corrected support-token count aggregation only. Canonical amended runs are in runs/dev_51601 and dev_51602. The original sweeps remain under the two pre-amendment directories. Five tests passed. Deterministic replay verified all 22 nonempty payload hashes, task splits, extracted vectors and metrics; max metric difference was zero. Fresh 51611–51613 were not opened.

## D — FAIL

Explicit function vectors increased held-out accuracy from .156 to .219 in seed 51601 and .156 to .188 in seed 51602; mean gold-candidate logprob improved by 2.97 and 2.62 nats. The frozen learnability gate therefore passes on the logprob criterion in both worlds.

Rank-4 Mirror accuracy was .188/.219, within .05 of explicit (.219/.188). But gold-candidate logprob was -11.446/-11.113 versus -10.840/-10.577 explicit, a loss of .606/.535 nats, exceeding the frozen .10 limit. Its actual uncompressed payload was 12,688 B vs 34,454 B explicit FV (0.368x, 63.2% fewer intervention bytes); full deployment was 168,157,312 B vs 168,179,078 B because the shared model dominates. The operation proxy was 139,264 vs 131,072 for explicit, plus held-out scoring times 2.02/2.99 s vs 2.04/2.04 s; timing varied across repeated CPU runs, so no runtime win is claimed.

Native centered PCA matched every corrected Mirror rank in bytes and exact query metrics. Thus the measured compression is native PCA, while rank4 also misses the gold-logprob preservation gate. FAIL; fresh remained sealed.

## C — Strongest counter-hypothesis

The four basis-heldout tasks are translation directions, while the shared basis is fit on country/state/element/number/animal relations. Their function vectors may occupy directions not represented at rank4. Increasing rank restores some held-out metrics but approaches explicit storage. Any benefit is fully reproduced by native PCA.

## U — Scope limits

One 70M Pythia model, 16 relation directions and constrained ranking over eight candidate answers are a narrow mechanism screen, not broad language-quality evidence. This is an approximate support-derived residual-stream vector, not an exact replication of the paper's attention-head extraction. No open-ended generation, user preference, semantic generalization beyond these relations, or production latency was established.

## Evidence classification

- **Facts:** model/revision hashes, support/task splits, held-out scores, actual NPZ bytes, total deployment bytes, compute, timings and replay are retained in freeze/amendment records, runs/, RESULTS_CORE.csv and VERIFICATION.json.
- **Interpretation:** explicit task vectors have a measurable held-out logprob signal, but rank4 loses too much of it; higher ranks and compressed points are exactly ordinary PCA.
- **Hypothesis:** richer task distributions, layers or task-aware nonlinear codes may yield a different frontier; these data do not establish that.
