# MA-393 — Adaptive-capacity embedding + Mirror view

Status: FAIL (development screen; fresh sealed). Base commit: `76ef3a5474ca1ad0e82600e3c080bbd831c8da70`. Prior art: PA60.

## Hypothesis

A token-specific low-description Givens coordinate may restore useful rare-token function diversity over 4D adaptive embeddings at less storage than simply increasing rare-token dimension.

## Direct controls

Compare full 16D tables, native adaptive 16D common/4D rare embeddings, equal-byte native rare-width allocation, per-token Mirror angles, and a per-token scalar gate. Rare tokens receive eight times fewer training examples than common tokens. Every coordinate, projection, classifier, metadata field, and archive byte is charged.

The frozen gates and seeds are in `PROTOCOL.json`. The teacher is deliberately aligned to factorized Mirror structure, so results remain a narrow synthetic mechanism screen; no capacity or natural language claim follows from role count alone.

## Results

Full and adaptive-native rare-token accuracy saturated at 0.99–1.00. Mirror reached 0.938 in one world but never improved over adaptive4 by the required 5 points. Its payload was 1.050x adaptive5 and throughput 0.858–0.862x the fastest adaptive native control. Fresh seeds remain sealed. See [REPORT.md](REPORT.md), [RESULTS_CORE.csv](RESULTS_CORE.csv), and [VERIFICATION.json](VERIFICATION.json).
