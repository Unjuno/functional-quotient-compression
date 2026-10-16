# MLP-ROLE-001 — Role-specific Mirror MLP use map

**Status: CPU experiment completed (2026-10-17); best native task readout beats Mirror.** [Research decision](../../../../docs/phase2/MLP_MIRROR_USAGE_DECISION_2026-10-17.md).

Mirror insertion: Add two independent low-description Givens codes to the gate and up preactivations of a shared SwiGLU, yielding 2×2 logical feature functions from one shared set of large gate/up/down weights. This is a *dense four-view* comparison, not sparse routed MoE.

Primary 4 roles ×3 worlds ×2 initializations ×7 methods = **168 trained models** at 550 AdamW updates, batch64, LR0.012 selected in earlier development (56 calibrations). Another 4×3×2×6 = **144** fits used separate worlds with 1,400 updates. Both protocols source-locked before opening respective fresh worlds, with main fresh fixed-final metric vs best validation checkpoint as secondary. All 624 final/best inference payloads decoded and independently checked, **9 tests pass**, eight deterministic full retrainings reproduce payload SHA/metrics.

Roles:
* Digits attribute classification: Mirror BCE **0.11308**, equal-parameter gain **0.13772**, native shared FFN with task-specific readout **0.10321** (smaller and faster). Three worlds.
* Digit image-patch denoising: Mirror MSE **0.03676**, gain **0.03570**, task-specific readout **0.03292**.
* Diabetes transformed target regression: Mirror **1.2630**, gain **1.2597**, readout **1.3229** (four *transforms of one* real label).
* Four-step synthetic autoregressive forecast: Mirror **0.42973**, gain **0.42924**, readout **0.43001**.

The models work on one common input with four logical heads. Independent MLPS are a quality reference, not a matched-byte comparator. No natural language, GPU, cache, or multi-corpus validation. Raw histories, code, source lock and model payloads in conversation's `MLP_MIRROR_USE_20261017_FULL_EVIDENCE.zip`; a source-only reproducible archive is separately provided. The GitHub folder contains the audited compact research result, not a claim of all raw payloads being committed. Existing MA ledger and main unchanged.