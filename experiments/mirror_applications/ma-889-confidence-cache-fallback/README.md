# MA-889 — confidence-gated cross-model KV translation

Status: **FAIL**
Evidence lane: CROSS_MODEL_KV / RELIABILITY / QUALITY / STORAGE / COMPUTE
Branch: `research/ma-889-confidence-cache-fallback-20261112`
Base: `480cac39b`

## Hypothesis

A source-only KV RMS novelty score can identify unsafe pairwise cache translations and fall back to native target prefill while preserving target quality with limited fallback.

## Frozen experiment

See `PROTOCOL.json` and `FROZEN_CONFIG.json`. Two source-target pairs were evaluated: Pythia-70M to Pythia-160M and SmolLM2-135M. Fresh article buckets used seeds 12889, 12893 and 12899 and were reserved before model scoring.

## Result

FAIL: Pythia-160M fell back on all prompts. SmolLM2 mean regret was +1.277 nat and missed the +0.5 gate in two of three fresh buckets. The paired fanout latency beat two native target prefills in all three buckets. Gate state added 30 B to 7.79 MB of full per-pair ridge maps; no mapper compression was achieved.

See `RESULTS.md`, `RESULTS_CORE.csv`, `SUMMARY_METRICS.json`, `PAYLOAD_INVENTORY.json` and `VERIFICATION.json` for the full H/T/D/C/U record and provenance.
