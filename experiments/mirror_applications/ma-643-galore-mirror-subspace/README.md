# MA-643 — GaLore Mirror subspace codes

Status: FAIL. Evidence lane: synthetic optimizer-state mechanism.

## H

For 16 task matrices whose gradients reuse K=1/2/4/8 rank-4 left subspaces, one shared projector dictionary and per-task Mirror codes can reduce GaLore resume-state storage while retaining held-out fit.

## T

Compared full Adam, independent per-task GaLore projectors, shared dictionary selection using initial-gradient alignment, and a native projector-ID control. Two development seeds; 64 examples and 100 optimizer updates per task. Actual inference payload, full resume payload, and geometry-only serialized bytes were measured.

## D

At K=1/2/4, full resume payload is 91.9%/92.5%/93.6% of standard GaLore; geometry-only state is 16.7%/22.5%/33.9% of independent projector bytes. At K=8, these ratios rise to 95.8% and 56.8%. K=4 seed 64302 has held-out MSE 1.117x standard GaLore and misses the 1.10x gate. Mirror and native projector-ID outputs and saved states are identical, so the result is a generic projector-sharing signal, not Mirror-specific evidence. Fresh seeds remained sealed.

## Fact / Interpretation / Hypothesis

- **Fact:** 32 development rows; three tests pass; full payloads and metrics replay.
- **Interpretation:** The projector dictionary cuts geometry storage, but low-rank Adam moments and model weights limit complete checkpoint savings.
- **Hypothesis:** Natural optimizer workloads may have recurring projector structure, but it must outperform ordinary projector IDs and retain quality at matched compute.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `source/development.json`, and `VERIFICATION.json`.
