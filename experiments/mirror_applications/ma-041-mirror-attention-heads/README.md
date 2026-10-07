# MA-041 — one QKV to multiple Mirror heads

Status: SCREENING. Branch `research/ma-041-mirror-attention-heads-20261007`.

## H

One physical QKV projection plus per-head input-space Givens coordinates may recover four useful heads from a shared-view teacher, using fewer serialized bytes than ordinary MHA. Tying, gains, and rank-1 residuals are controls.

## T

Synthetic 16D full-attention block (sequence length 8, four heads of dimension 4), shared output projection. Compared independent-QKV MHA, hard tying, per-head projection gains, rank-1 residuals, and Mirror views. Second teacher uses independent QKV projections. Development world 41000 selected common LR 0.003; fresh worlds 41001–41003 remain sealed. Source/protocol/tests/selection/dev-row hashes are frozen in `FREEZE_MANIFEST.json`.

## Development screen

Aligned output MSE at LR 0.003: full MHA 7.05e-5; Mirror 8.62e-5; rank-1 residual 5.34e-4; tied QKV 1.61e-3. Actual serialized payload: 4,065B Mirror vs 6,053B full MHA (67.1%). The all-independent teacher remained hard for all shared methods. These are development-only observations.

## Decision

FACT: fresh verification pending.
INTERPRETATION: dev shows a storage-quality signal, but selected-LR Mirror was 1.22x full MHA MSE.
HYPOTHESIS: fresh worlds may or may not reproduce the improvement over tying.
BOUNDARY: synthetic attention only; no language NLL claim.
