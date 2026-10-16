# MTPM-001 — Mirror multi-token drafting (2026-10-17)

**Status: COMPLETED EXPLORATORY SYNTHETIC MECHANISM, NOT ADOPTED.**

[Full Japanese report](../../../../docs/phase2/MTPM001_MIRROR_DRAFT_RESULTS_2026-10-17.md) records all primary facts and limitations. [Primary median CSV](PRIMARY_WORLD_MEDIANS.csv) and [prospective longer-budget follow-up CSV](FOLLOWUP_WORLD_MEDIANS.csv) are summaries of recorded evidence, not raw per-seed results.

This study trains a shared **causal** three-token prefix transformer to produce four speculative draft tokens. Mirror inserts a low-description Givens code into a shared future-output head. Hidden-branch mode has two unsupervised latent packet plans. The cross-direction-attention variant only mixes latent horizon slots within each plan, not all plans and tokens.

**Primary:** Three independent hidden random-rule worlds × two seeds, 900 AdamW updates, minibatch 64. Mirror median full joint packet NLL 0.912 and true oracle-target p/q acceptance 3.795 of four tokens per proposed packet at 46,192 serialized bytes. Independent 2-plan MTP heads: 2.496 NLL, 2.343 accepted, 60,216 bytes. A simple **native gain** is competitive: 1.104 NLL, 3.641 accepted, 46,702 bytes. Cross-direction attention does **not** pass its 0/3-world marginal-value gate. Revealed cyclic Mirror-vs-independent byte/quality gate fails 1/3 worlds; revealed random passes 2/3, not a general claim.

**Separately prelocked LONG follow-up:** Three **new** random-hidden worlds, two seeds and four methods, 2,400 updates each; Mirror NLL is lower than independent MTP in 3/3 worlds, better than gain in 2/3; attention variant remains unstable.

**Critical caution:** This is a finite-state IID toy model with an exact oracle-target verifier, NOT natural-language speculative decoding, a real LM target, full rejection-correction/residual sampling, KV rollback, actual decode throughput, GPU/fused kernels, or equal wall-time learning. Inference speed advantage is **NOT** established, and previous MA-249 and TM001 statuses are unchanged.

## Reproducible evidence packaging

The conversation's downloadable **MTPM001_SOURCE_20261017.zip** contains model.py, train.py, run.py, batch_launch.py, verifier.py, audit.py, long_run.py, long_audit.py, unit tests, original preregistered PROTOCOL.json and both SHA-256 source locks, result CSVs and full report. SHA-256: `20be41eff0aa77d25edad9e1f43dc34d14204d4869574d1846ca7282fd75516c`.

The conversation's downloadable **MTPM001_FULL_EVIDENCE_20261017.zip** contains the source and 156 serialized inference payloads, per-run JSON, p/q audits and runtime logs. SHA-256: `8be16bbb8d263caee4cec8d667ed58edd14005ddda82d4ad9f85eb3c13b02cad`.

**These ZIP files are conversation attachments, not GitHub blobs.** This branch contains the report, verified summary results, and reproducibility manifest. Obtain the exact scripts and full evidence from the attached ZIPs; do not assume all training scripts were committed here. No main merge or canonical MA registry update.
