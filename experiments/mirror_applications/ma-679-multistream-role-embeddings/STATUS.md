# MA-679 status

- Status: FAIL (Mirror-specific claim)
- Branch: `research/ma-679-multistream-role-embeddings-20261104`
- Development: seeds 67901/67902; protocol amendments recorded before corrected rerun
- Fresh/audit: sealed after exact native control alias

## H — Hypothesis

Four isolated logical streams can share a common embedding table plus rank-1 role atom and scalar code, retaining held-out token embeddings, reducing complete bytes, and preserving zero cross-stream leakage. Mirror must beat native rank-1 role-conditioned embeddings.

## T — Treatment and controls

Synthetic vocabulary 64, width 16, four streams, two development seeds; measured 24 held-out token IDs per stream. Compared independent stream tables, hard tying, shared base plus rank-1 role code, and native shared-basis role coefficients. Cross-stream probe perturbed all non-target stream token IDs while holding the target fixed. The initial lookup-index and cumulative perturbation bugs are preserved under `source/attempts/`; amendments preceded the corrected metrics. Tests replay outputs and leakage from serialized payloads.

## D — Decision

FAIL for Mirror-specific value. In both seeds, rank-1 role codes reproduce held-out embeddings exactly and have zero measured cross-stream leakage. Payload is ~8.3KB versus ~15.4KB independent (about 54%). Hard tying is smaller (~4.1KB) but loses role quality (NRMSE .44/.74). Native rank-1 control has exactly the same function/compute and is 16B larger due to serialized key naming. Fresh stayed sealed.

## C — Strongest counter-hypothesis

This is ordinary low-rank factorization of stream-specific embedding deltas. Native role-conditioned basis codes account for the storage gain exactly; independent stream identity does not imply Mirror-specific functionality.

## U — Unconfirmed

No learned multi-stream transformer, parallel causal attention, training, natural stream task quality, or hardware latency is tested. The leakage probe verifies this table lookup's isolation only.

## Evidence labels

- Fact: held-out embedding errors, payload bytes, lookup MAC proxy, wall time and leakage replay are in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: aligned stream-role embedding deltas compress at this synthetic rank-1 screen without cross-stream contamination.
- Hypothesis: Mirror-specific stream views would need to improve over native role embedding factorization inside a trained multi-stream model.
