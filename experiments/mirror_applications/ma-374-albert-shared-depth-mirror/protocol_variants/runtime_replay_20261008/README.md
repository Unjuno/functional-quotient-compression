# MA-374 A1 runtime replay audit (2026-10-08)

## Why this variant exists

The checked-in development JSON contained payload sizes and SHA-256 values, but the corresponding twelve `.npz` inference payloads were absent from the canonical branch. Re-running the frozen development source and seeds under the available runtime did not reproduce those hashes. This variant preserves the original result unchanged and records what can be reproduced in the current environment.

## Frozen boundary

- This is a same-seed, same-source replay audit, not hyperparameter tuning.
- The original seeds are 37401 and 37402; no fresh seed was opened.
- The original preregistered multi-depth gate remains missed; fresh seeds 37411, 37412, and 37413 remain sealed.
- No model architecture, optimizer, update count, data generation, or metric was changed.
- Current source commit: `4e75142d1848ad4e7f51de9fa1dbda0bd02990a7`.
- Runtime: Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.3.5, Linux x86_64.

## Replay outcome

FACT: all twelve current-runtime payloads are archived here and replay exactly against their current-runtime JSON metrics and hashes using the experiment verifier. Their sizes remain 40,821 B (untied), 11,889 B (ALBERT tied), 31,209 B (attention-shared), 25,077 B (FFN-shared), 12,131 B (scalar), and 12,131 B (Mirror).

FACT: none of the twelve current-runtime hashes matches the hash in the original JSON. Some metrics also drift materially. For example, seed 37401 FFN-shared validation NLL is 0.178127 in the original record and 0.210332 in this replay; seed 37402 untied validation NLL is 0.147938 versus 0.201633.

INTERPRETATION: the original exact-replay verification cannot currently be independently reproduced from the canonical branch, because its payload artifacts and runtime provenance were not retained. The current-runtime variant is reproducible, but it does not retroactively validate the original artifacts.

BOUNDARY: this audit does not open fresh data or select a new configuration. The scoped development-only final-depth signal remains PROMISING, while the registered multi-depth gate remains failed and no fresh/capacity claim is established.
