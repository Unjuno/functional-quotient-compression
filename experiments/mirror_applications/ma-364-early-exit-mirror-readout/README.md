# MA-364 — Early-exit Mirror readout views

Status: SCREENING; frozen before development
Branch: `research/ma-364-early-exit-mirror-readout-20261008`
Base: `c935a90`
Prior art: PA50 DeeBERT / Depth-adaptive Transformer.

## H — Hypothesis

Three depth-specific logical readouts can be represented as one shared readout matrix plus compact Mirror depth coordinates, preserving exit-wise prediction quality and the same early-exit compute policy at lower serialized head bytes than separate physical readouts.

## Mirror insertion

> **Mirror insertion:** this experiment adds one depth-specific scalar coordinate to a shared linear classifier so it can serve as logical readout heads at multiple exit depths without storing a full head per exit.

Controls: three independent exit heads; one hard-shared readout; shared readout plus ordinary direct coefficient basis; shared readout plus Mirror depth coordinates. Exit assignments are fixed identically for all methods; no policy is tuned per model.

## T — Frozen protocol

Synthetic 64D hidden states at exits 2/4/6, 8 classes, 4,096 examples per world. Teacher heads are drawn from a shared affine readout basis plus depth coefficient; target labels sampled from exit-specific softmax. Fixed exit assignment uses a precomputed difficulty rank: 50% exit at depth2, 30% at depth4, 20% at depth6. Dev seeds 36401/36402; fresh 36411/36412/36413 locked. Methods serialize independent heads, hard-shared head, direct coefficient basis, or Mirror coordinate representation. Evaluate NLL/accuracy for each exit and under the fixed mixed-exit policy, actual head payload bytes, and average dense MACs. This is an oracle mechanism screen with no training updates.

## Gates

PASS both dev seeds if mixed-policy NLL within 0.01 of separate heads, per-exit NLL within 0.02, >=10% head-byte saving, and identical average MAC count; direct coefficient control must be >=10% larger for Mirror-specific claim. FAIL if quality misses or direct control is within 10%. Fresh stays sealed on dev failure.

## Boundaries

Synthetic hidden states/readout bank only, no trained Transformer or adaptive exit policy. This tests head parameter sharing; early-exit compute benefits are fixed by the common policy and are not attributed to Mirror.

## Result — development

**FAIL for Mirror-specific value.** Across both development seeds, Mirror and direct coefficient controls had identical per-exit NLL/accuracy and mixed-policy results; payloads were also byte-equal (4,554/4,565B), with metadata-only hash differences. Independent head bank was 6,061/6,071B with equal prediction quality. Hard sharing was smaller (2,270/2,276B) but had worse mixed NLL by ~0.04. All methods used identical exit fractions and average MACs (59.14M). Fresh seeds remain sealed because direct coefficients matched exactly.

**Fact:** ordinary shared-basis coefficients retained exact per-exit outputs; Mirror and direct controls had equal byte counts and predictions.
**Interpretation:** head sharing can reduce bytes relative to independent exits, but scalar depth coordinates are ordinary coefficients and no adaptive compute benefit is attributable to Mirror.
**C:** the oracle task's heads lie exactly on a one-dimensional affine codebook, which favors sharing.
**U:** trained early-exit policy, language modeling, adaptive routing, and fresh worlds are untested.
