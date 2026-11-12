# MA-879 — Attention-sensitive calibration of shared KV bridge codes

Status: SCREENING. Prior art: PA237 (CacheBridge) and PA227 (Fisher-geometric sensitivity). Base: MA-878 cumulative evidence state.

## Hypothesis

On the fixed Pythia-70M→SmolLM2-135M pair and fixed [0,4,7] source-head support, calibrating rank-32 shared-basis KV-map coefficients through the target model's causal next-token loss will improve held-out target NLL over the same unoptimized PCA codes, while retaining at least a 50% serialized-byte reduction versus independent attention-weighted ridge maps.

## Mirror insertion

> **Mirror insertion:** a small per-target-role coordinate `m` selects affine cache maps from a shared rank-32 map basis; calibration updates `m` against target autoregressive loss while physical source/target model weights and the shared basis stay fixed.

- Native baseline: target's own prefill cache.
- Existing method: attention-weighted affine ridge map per target layer/head/KV role.
- `m`: one rank-32 coefficient vector per target layer/head/KV role, serialized as float32.
- Cheapest ordinary control: project ridge maps into a PCA basis and directly optimize the same low-rank coefficients by target NLL. This is a native shared-basis/adaptation control and may fully explain any result.
- Support: fixed [0,4,7] source KV heads for every target layer. This isolates the calibration objective from MA-878's head selection.
- Position: strip source RoPE, map key content, and apply target RoPE; values are mapped without rotation, as in MA-877/878.

## Comparisons

1. Native target prefill.
2. Full independent attention-weighted ridge maps (CacheBridge-style control).
3. Rank-32 shared PCA reconstruction of those maps with unmodified projected codes.
4. Same shared rank-32 basis with per-role codes optimized by target next-token NLL.

The shared-basis code is also the ordinary low-rank coefficient control; equality to that control means no Mirror-specific gain.

## Frozen protocol

- Pair: EleutherAI/pythia-70m-deduped → HuggingFaceTB/SmolLM2-135M.
- Dataset: Salesforce/Wikitext-2 raw.
- Dev: train split, seed 829, 24 calibration prompts and 16 held-out prompts.
- Fresh: validation split for 12 calibration prompts and test split for 16 test prompts per seed, seeds 983, 997, 1009. Exact tokenizer offset alignment and deterministic disjoint row selection.
- Basis rank: fixed at 32; no rank search after this protocol freeze.
- Code optimization: Adam, 24 Adam updates per development learning-rate candidate (one pass over 24 dev calibration prompts); candidates `{1e-4, 5e-4, 2e-3}` selected by held-out dev NLL. Fresh runs use the selected learning rate and exactly 24 updates (two passes over 12 calibration prompts).
- Each update uses one calibration prompt: transfer its source prefix cache, feed the target query token, and minimize the target's next-token cross-entropy. Target model weights are frozen. No test prompt participates in code fitting.
- Prompt target query is the penultimate target token and label is the final target token; transferred cache is truncated to the preceding prefix.

## Gates

**PASS for this scoped application:** in all three fresh seeds, optimized rank-32 codes improve held-out NLL over unmodified rank-32 PCA by at least 0.10 nat, stay within +0.50 nat of native target NLL and +0.50 nat of full ridge NLL, and the actual serialized rank-32 payload is at most 50% of full-map payload.

**FAIL:** two or more fresh seeds miss any required quality gate, or the byte gate fails in two or more seeds, or prompt/serialization correctness checks fail.

**Mirror-specific boundary:** even a scoped pass is not a Mirror-specific result if ordinary direct optimization of the same low-rank coefficients matches it; that is reported as a native shared-basis/adaptation alias.

## Accounting

Actual serialized payload bytes include the shared mean, basis, all per-role codes, role/layer/head references, model-pair identifier, and format header. Model weights are common and reported separately. Report prompt/token counts, 24 optimizer updates, layer-token forward proxy, basis-fit and gradient-calibration time, and CPU cache application time. No GPU/runtime serving claim is planned.

## Scope

One ordered model pair, one natural-language corpus, three fresh partitions, one-token autoregressive probes, CPU only. No generation-trajectory or CacheBridge paper-reproduction claim.
