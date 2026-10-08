# MA-405 — StyleGAN2 modulation versus Mirror FFN Views

Status: **FAIL at the frozen development screen**.
Prior art: PA65 (StyleGAN2 weight modulation and demodulation).

## H — Hypothesis
A compact non-diagonal Mirror View should provide an efficient logical FFN when context variation is rotation-dominated, complementing StyleGAN2's diagonal per-channel modulation. On the StyleGAN2 endpoint, its native control should remain strongest.

## T — Treatment
A shared frozen 32D nonlinear feature block and output projection `W` were used for eight contexts and five target mixtures. Alpha 0 is exactly `W @ R(m)`; alpha 1 is exactly StyleGAN2 channel modulation/demodulation; the middle targets linearly combine the two effective matrices. Five controls were run: unmodulated shared FFN, StyleGAN2 modulation, Mirror rotation, rank-one residual, and independent full affine matrix. Each development seed used 4096 train and 4096 separate held-out inputs per context/alpha, 400 updates, batch size 1024. Fresh seeds 40511–40513 stayed sealed. Actual compressed NPZ bytes include shared tensors, all logical task state, and metadata.

## D — Decision: FAIL
At alpha 0, Mirror was exact to about `5e-4` NRMSE and substantially beat StyleGAN2, but the screen required both seeds to meet every storage and runtime gate. Seed 40501 missed the throughput threshold.

| Seed | Mirror NRMSE | Style NRMSE | Mirror / Style / independent bytes | Mirror / Style throughput | Gate |
|---|---:|---:|---:|---:|---|
| 40501 | 0.000447 | 0.636037 | 8,199 / 9,381 / 82,699 | 19.79M / 31.27M (0.633x) | FAIL |
| 40502 | 0.000473 | 0.548260 | 8,214 / 9,367 / 82,748 | 17.98M / 18.06M (0.996x) | PASS |

Mirror payload was 0.874–0.876x StyleGAN2, passing the 0.95 limit, and 0.0992–0.0993x independent full, just within the 0.10 limit. Seed 40501 throughput was below the frozen 0.80x threshold, so the combined two-seed gate failed and no fresh test proceeded.

At alpha 1, StyleGAN2 reaches NRMSE 0.00043–0.00045 while Mirror is 0.464–0.494. Across mixture points, both structured controls degrade; the independent matrix stays around 0.002–0.007 NRMSE. These mixtures show private full-matrix state is needed for these composite functions.

## C — Strongest counter-hypothesis
The alpha 0 target was constructed from the same Givens family used by Mirror, while alpha 1 was constructed from StyleGAN2. This isolates endpoint expressivity but does not establish natural FFN utility. The byte advantage over StyleGAN2 is about 12.5%, and runtime was inconsistent across development seeds.

## U — Unknown
Natural Transformer language quality, robust runtime on optimized kernels, and fresh replication remain untested.

## Fact / Interpretation / Hypothesis
**FACT:** Mirror accurately represented the rotation endpoint with fewer total bytes than StyleGAN2; StyleGAN2 accurately represented its native endpoint. The combined screen failed the runtime gate in one seed. Serialized replay passed for all 50 method/seed/alpha rows. Fresh remained sealed.

**INTERPRETATION:** The experiment shows complementary structured families, not general replacement: Givens Views capture non-diagonal variation, StyleGAN2 captures diagonal modulation, and intermediate mixtures require much more private state.

**HYPOTHESIS:** Combining StyleGAN2 scale codes with Mirror rotation may cover a broader function family, but such composition is a separate MA-407-style experiment, not established here.
