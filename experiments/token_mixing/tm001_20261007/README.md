# TM001 — Parallel Period Token Mixing

Synthetic mechanism test for emitting P future tokens in one forward.

Files:
- tm001.py: tiny Transformer, KV-cached AR, direct period slots, triangular period mixer, packet-latent diagnostic, deterministic serializer.
- engine.py: training, exhaustive evaluation, cached generation, benchmark helpers.
- test_tm001.py: regression tests.
- PROTOCOL.json: frozen experimental scope and evidence boundaries.
- RESULTS_CORE.csv: checked core results.
- BENCHMARK_SPEEDUP.csv: CPU period/cached-AR throughput ratios.

Main result: P-token parallel decoding is viable when packet-determining information is already present in the context. A hidden packet-level branch variable breaks factorized phase-slot decoding; triangular slot attention alone does not solve this.

Narrative report: ../../../docs/phase2/TM001_PARALLEL_PERIOD_TOKEN_MIXING.md

The complete local evidence bundle contains all trained payloads, raw JSON metrics, logs and SHA-256 manifest; large model artifacts are intentionally not committed to ordinary Git history.
