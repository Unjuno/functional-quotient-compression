# MA-886 — one producer KV cache fanned out to two targets

## Decision

**FAIL** under the preregistered joint quality, latency, and system accounting. A single pooled translator plus target-role codes makes a compact serialized mapper, and its summed handoff latency beats native prefill in two of three fresh buckets. The translated caches miss the native next-token quality gate in all six target×seed conditions. The independent per-role ridge control is much closer to native but remains slightly above the +1 nat limit on 4/6 conditions. The source checkpoint is an added 166 MB system object.

## H — hypothesis

One Pythia-70M producer prefill, translated through one shared affine map and compact per-target-role gain/bias coordinates, would provide useful cache states to Pythia-160M and SmolLM2, keep target NLL within +1 nat of native, beat two native target prefill paths in summed wall time, and use no more than 25% of the independent per-role ridge mapper bytes.

## T — experiment

Producer: EleutherAI/pythia-70m-deduped. Targets: EleutherAI/pythia-160m-deduped and HuggingFaceTB/SmolLM2-135M. Target cache role counts were 288 and 180. Nearest-relative layer/head alignment, inverse RoPE before fitting and target RoPE restore before replay. One pooled 65×64 affine map was fit over both targets and all target roles; target gain/bias coordinates were calibrated per role. Controls were target-global affine per architecture, independent affine per target role, and each target’s native prefill plus cache replay.

Development used bucket 0 (24 fit / 8 check / 16 test). Settings were frozen before fresh access. Fresh seeds 10883/10889/10897 used separate whole-article buckets 1/2/3, also 24/8/16. Prompt text and title hashes were excluded from MA-883 and Wikitext-2; source and both target tokenizers had exact character-offset alignment. All 6 fresh target×seed rows are retained in `RESULTS_CORE.csv` and `fresh_results.json`.

CPU only; CUDA unavailable. Fresh cache preparation took 86.8s total; map fitting took 144.6s. Maximum RSS was 3813 MiB. Checkpoint SHA-256 and serialized mapper hashes/bytes are recorded in machine-readable results.

## D — result

Status: **FAIL**.

**Facts.** Mean native / shared-view / target-global / full-ridge NLL was:

- Pythia-160M: 3.386 / 6.645 / 11.723 / 4.532 nat.
- SmolLM2: 2.463 / 8.648 / 11.325 / 3.571 nat.

The candidate passed native+1 nat in 0/6 target×seed rows. Its summed source-prefill-once + both mapper applications + target replays beat the two native prefill+replay paths in 2/3 seeds: 10883 0.610s vs 0.751s, 10889 0.469s vs 0.513s, 10897 0.487s vs 0.440s. No quality/latency warmup or GPU claim is made.

**Bytes.** Candidate mapper payload was 138,560 B. The independent full-ridge mapper bank was 7,787,752 B; candidate is 1.78% of it and passes the mapper-only byte gate. Target-global mapper was 33,515 B. Producer checkpoint is 166,029,852 B; the two target checkpoints total 644,059,248 B. Native two-target service uses those target checkpoints; fanout adds the producer checkpoint, yielding a combined model+mapper system payload of 810,227,660 B, versus 644,059,248 B for target-only native service (+25.8%). The mapper is small; the required producer is not free.

The shared mapper proxy is 1,976,832 MACs per fanout prompt token across both target roles. Serialized payload round-trip replay max absolute NLL delta was exactly 0.0 nat for all three methods and all seeds. Actual bytes and SHA-256 values are in `PAYLOAD_INVENTORY.json`.

**Interpretation.** Compact role codes materially improve on one target-global affine matrix but do not bridge the cross-model functional gap to native targets. Independent per-role ridge substantially improves NLL and is faster in this CPU harness, but it is ~7.8 MB and still misses the strict +1 nat limit on four rows. The candidate’s fanout latency gain is not robust across the three seeds.

## C — strongest counter-hypothesis

The experiment measures one-token replay and fits on only 24 calibration prompts per target. Longer calibration, target-side context correction, layer-specific nonlinear maps, or longer decode dwell could improve quality or amortize producer cost. Those approaches add state/compute and were outside this frozen experiment; no post-fresh adjustment was made.

## U — unresolved

No multi-token rollout, long-context/1K/4K cache, target-side correction model, confidence fallback, GPU fused application, energy, or cross-family target sweep was run. Producer checkpoint cost could be amortized if it already serves other jobs, but that serving scenario was not tested. This is not a CacheBridge or PA239 reproduction.

## Fact / interpretation / hypothesis separation

- **Fact:** per-condition NLL, actual translator bytes, SHA-256 hashes, model checkpoint sizes, and timed source-once fanout are stored in the checked JSON/CSV artifacts.
- **Interpretation:** this shared global map does not meet cross-target quality; CPU fanout speedup appears in only two of three fresh buckets.
- **Hypothesis:** native-compatible or context-corrected targets may improve the quality frontier, at additional cost; untested.
