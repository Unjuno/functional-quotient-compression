# MA-582 — MLA latent plus layer Mirror reconstruction

Status: SCREENING  
Branch: `research/ma-582-mla-layer-mirror-reconstruction-20261009`  
Base commit: `6574132b`

## Hypothesis

**H:** A rank-128 MLA latent basis shared across each fixed three-layer group, with a rank-4 layer-specific residual View, can recover per-layer K/V behavior while lowering actual serialized state versus per-layer MLA and FP16 caches.

**Mirror insertion:** `m` is the per-token, per-layer, per-K/V-role coefficient vector decoded through a group-shared residual dictionary. Physical PCA and residual bases are shared across layers; logical K/V roles retain distinct codes.

Prior art: PA115 (MLA) and PA08 (MLKV). The native control uses the same group-shared residual dictionary and code payload to test whether Mirror provides anything beyond ordinary shared-basis coding.

## Comparisons

FP16 full KV; per-layer MLA rank 128; group-shared rank 128; group-shared rank 128 plus rank-4 layer Views; exact native group-shared residual coding; and per-layer private residual upper control. We score 16 one-token queries after 64-token prefixes using Pythia-70M and WikiText-2. Four train prefixes fit all bases. Dev seeds are 58201/58202; fresh test seeds 58211–13 remain sealed unless all dev gates pass and no native alias occurs.

## Decision

Results are recorded in `RESULTS.md` and `RESULTS_CORE.csv` after development. Report separates facts, interpretation, and hypothesis, and includes H/T/D/C/U.
