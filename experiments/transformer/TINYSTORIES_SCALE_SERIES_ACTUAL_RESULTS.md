# TinyStories GPT-Neo scale-series actual results

This note records high-value **actual-model** experiments run locally in the container on the user-supplied `1M.zip`, `3M.zip`, `8M.zip`, and `28M.zip` checkpoints. The experiments did not use GitHub Actions as the compute runner.

## Scope and evidence boundary

All four checkpoints are GPT-Neo-family models with 8 transformer layers, 16 attention heads, vocabulary size 50,257, and hidden sizes 64 / 128 / 256 / 512 respectively. A local GPT-2 BPE tokenizer and a manual PyTorch GPT-Neo forward were used for fixed short-context probes. These results are real-model evidence for these checkpoints, not SmolLM2 evidence and not a 64x certificate.

The reported `bps` values below are embedding-component bits per original embedding scalar for the tested representation. They are not whole-model compression ratios.

## 1. Scale changes where the parameters live

Counting floating checkpoint tensors while excluding causal-mask buffers gives:

| model | float scalars | embedding fraction | attention fraction | MLP fraction |
| --- | ---: | ---: | ---: | ---: |
| 1M | 3,745,984 | 85.86% | 3.51% | 7.07% |
| 3M | 8,278,400 | 77.71% | 6.35% | 12.73% |
| 8M | 19,702,528 | 65.30% | 10.65% | 21.34% |
| 28M | 51,987,968 | 49.50% | 16.14% | 32.31% |

Thus embedding dominates the smallest checkpoints, while attention/MLP become progressively more important with scale.

The actual embedding spectrum also becomes less concentrated with scale:

| model | hidden size | top-35 explained variance | effective rank |
| --- | ---: | ---: | ---: |
| 1M | 64 | 0.79896 | 50.2 |
| 3M | 128 | 0.53328 | 103.2 |
| 8M | 256 | 0.34734 | 207.5 |
| 28M | 512 | 0.21571 | 422.5 |

This invalidates a universal spectrum threshold for choosing an embedding codec. Actual rate overhead and hidden dimension must be included.

## 2. Uniform sub-bit embedding coding is the current bottleneck

A task-signature root plus Fisher-ranked private residual exceptions was evaluated on the 8M embedding with a fixed root and six unseen text probes.

| embedding rate | exceptions | mean KL | mean top-1 flip | mean NLL delta |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 bps | 2,214 | 0.8349 | 34.59% | 0.7185 |
| 0.50 bps | 5,260 | 0.5387 | 28.82% | 0.4430 |
| 1.00 bps | 11,351 | 0.3248 | 24.71% | 0.2614 |
| 2.00 bps | 23,535 | 0.0537 | 18.12% | 0.0482 |

For this codec family, 0.25 bps is not a near miss: even 1 bps still produces substantial functional change. Two bps is the first tested point with mean KL near 0.05, although top-1 changes remain material.

The larger 28M model is easier but does not make 0.25 bps automatically safe. On a matched four-probe comparison, a single calibration passage gave mean KL 0.9377 at 0.25 bps and 0.5316 at 0.5 bps. Replacing only the exception-importance estimate with five diverse calibration passages improved these to 0.8315 and 0.4495 respectively.

## 3. Calibration breadth matters most for the private exception selector

At 28M, keeping the root fixed and widening Fisher calibration from one passage to five passages changed the same-rate frontier substantially:

| rate | single-calibration selector | five-passage selector |
| ---: | ---: | ---: |
| 0.25 bps | 0.9377 mean KL | 0.8315 |
| 0.50 bps | 0.5316 | 0.4495 |
| 1.00 bps | 0.2741 | 0.2093 |
| 2.00 bps | 0.1457 | **0.03875** |

The 2 bps clinic probe fell from KL 0.3578 to 0.0449 after widening the selector calibration. This shows that apparent rate insufficiency can be confounded by calibration coverage, especially when many private exceptions are available.

Rebuilding the root itself on the richer calibration was much less important. At 0.25 bps it reduced four-probe mean KL from 0.8315 to 0.7970; at 0.5 bps it slightly worsened KL from 0.4495 to 0.4591. The large gain came from **which token rows received private bits**, not from simply making the root calibration larger.

On the 8M checkpoint, calibration importance ranks were highly correlated across passages (mean Spearman about 0.91), yet top-2,214 exception sets had only about 0.44 mean pairwise Jaccard overlap. Adding more calibration was not monotone: a nested 1-to-5 passage sweep reached its best six-probe mean KL at four passages (0.8183) and worsened at five (0.8349). A stable-core/median selector was worse than summing expected importance. Softly compressing extreme per-passage importance with a square-root aggregation gave only a small held-out improvement and is not treated as a major result.

## 4. Root/private allocation has a real hard-budget optimum

Using the rich exception selector at 28M and fixing the total embedding rate to 0.25 bps, larger roots were not better across four probes:

| root K | root bps | private exceptions | mean KL |
| ---: | ---: | ---: | ---: |
| 128 | 0.0251 | 2,781 | 0.8156 |
| **256** | 0.0373 | 2,631 | **0.7976** |
| 512 | 0.0596 | 2,355 | 0.8917 |
| 768 | 0.0819 | 2,079 | 0.9621 |

A single-probe experiment had previously preferred K=768, so the multi-probe result is an important correction: tuning root complexity on one probe can spend too many bits on the shared structure and starve private exceptions.

At K=256, changing private-residual precision while holding the total rate near 0.25 bps gave:

- 3-bit residuals: mean KL 0.7927;
- 4-bit residuals: 0.7976;
- 5-bit residuals: 1.0032.

The optimum is therefore discrete and rate-coupled; higher private precision can be worse because it reduces the number of protected token rows.

Weighting root centroids by the square root of Fisher importance produced a small additional improvement at K=256: mean KL 0.7976 -> 0.7815, despite worse weight NMSE. Full Fisher weighting was slightly worse than square-root weighting.

## 5. Tied input/output corrections should remain shared in the tested family

For the 8M tied embedding, the same stored private residuals were decoded with different role-application policies while accounting for one role-mode bit per exception.

Six-probe mean KL:

- apply each stored correction to both input embedding and LM head: **0.8350**;
- apply corrections only to the LM head: **2.6884**;
- apply input corrections only for the top 75% by input-gradient importance: 0.9218.

Thus role-specific decoded state did not help this representation. The tied correction itself has functional value and should not be split merely because the input and output roles are semantically different.

A separate selector experiment did find a small benefit from adding input-gradient importance to output Fisher importance (mean KL about 0.822 -> 0.802), but the selected set changed by less than 1%; output-side geometry remains dominant.

## 6. Implications for the current 64x program

These results do **not** prove that 64x is impossible. They do establish a strong negative result for the current embedding family:

- uniform weight-space PQ at about 0.23-0.25 bps severely damages all four tested model sizes;
- task-signature roots plus functional private exceptions are much better than uniform PQ, but the best tested 28M multi-probe point at 0.25 bps still has mean KL around 0.78-0.80;
- calibration breadth and task-aware bit allocation materially improve quality but do not close the 0.25 bps gap;
- larger models amortize shared-root fixed cost better, yet the embedding remains a first-order bottleneck.

The next codec work should therefore focus on a **different embedding representation**, not incremental tuning of the same root-plus-4-bit-residual family. Candidate directions include task-space predictive coding, structured vocabulary factorization, conditional/semantic roots with genuinely lower private entropy, or a decoder that derives token representations rather than storing near-independent row residuals.

## Boundaries

Do not infer that:

- the reported KL values are benchmark-quality measurements;
- 0.25 bps is impossible for all embedding codecs;
- the single best root K or residual precision is universal;
- improvements in KL imply improvements in every quality metric;
- individual component results can be added to predict whole-model joint distortion.
