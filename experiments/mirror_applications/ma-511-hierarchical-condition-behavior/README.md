# MA-511 — Hierarchical condition x behavior codes

Status: **FAIL for Mirror-specific attribution** (aligned held-out composition point passes its quality/byte gates).
Branch: research/ma-511-hierarchical-condition-behavior-codes-20261008
Prior art: PA101, Conditional Activation Steering.
Protocol freeze commit: bc853bbfd42654e3bf168039c58e50c1810fe0f2; test-only Amendment 1 recorded separately.

## H — Hypothesis

Fit 48 of 64 synthetic condition-behavior function vectors, withholding 16 whole pairs while preserving all condition and behavior IDs in the fit graph. A rank-4 condition code plus rank-4 behavior code should compose held-out outputs at relative RMSE <=.05 and use <=.50x the full explicit pair table bytes. To call this Mirror-specific, it must also beat native additive factorization by >=10% at equal quality.

Mirror insertion: independently encode condition and behavior factors in shared bases, then compose their decoded vectors for a requested pair. The pair query provides the two factor IDs to every method.

## T — Executed protocol

Two development seeds (51101, 51102), 64-dimensional outputs, 8 condition IDs, 8 behavior IDs, 48 visible pairs, and 16 held-out pair identities. Basis ranks {2,4,8}, rho {0,.1,.25} private pair residual. Amendment 1 changed only the unit-test tolerance from <1e-5 to <=.05 to match the preregistered quality gate; the initial failed assertion is preserved in runs/pre_amendment_1_test_gate_mismatch/. The runner and protocol were unchanged. Three tests passed. All 54 dev payloads replay byte/hash-identically; split manifests and metrics replay exactly. Fresh seeds 51111–51113 remain sealed.

## D — FAIL for Mirror-specific attribution

At rank4/rank4 and rho=0, held-out relative RMSE was .0244/.0452, with 16 distinct outputs, passing the <=.05 quality gate in both seeds. Actual payload was 4,136 B versus 16,932 B for the explicit 64-entry pair table (0.244x, 75.6% fewer bytes). The explicit CAST-style additive factors used 5,414 B and reconstructed the same targets at zero RMSE. The Mirror rank-4 decoder proxy was 576 ops/pair versus 192 for CAST additive and 64 for direct full-table lookup.

At rho=.1 rank4 held-out RMSE rose to .1115/.1200; at rho=.25 it was .2673/.2749. Native additive least squares plus PCA/SVD exactly matched Mirror output metrics and payload bytes at every rank/rho/seed. The standard additive composition point therefore explains the byte/function result completely. FAIL for Mirror-specific attribution; no fresh access.

## C — Strongest counter-hypothesis

The teacher functions are explicitly additive in condition and behavior. Ordinary two-factor regression recovers the same composition rule and PCA compresses it; the result is expected from the data-generating algebra. Independent pair-private directions cannot be inferred for unseen combinations and increase held-out error.

## U — Scope limits

This is a synthetic activation-function composition mechanism test, not a frozen language model or semantic conditional-steering experiment. Pair counts are not independent capacity; only 16 held-out outputs were measured. Natural-language utility, false triggers, human preference, language quality, and real model latency remain untested.

## Evidence classification

- **Facts:** held-out errors, byte counts, operation proxies, hashes and replay evidence are in RESULTS_CORE.csv, runs/ and VERIFICATION.json.
- **Interpretation:** factorized additive composition is a useful synthetic storage/compute tradeoff, but native additive/PCA fully explains it and direct lookup/addition offer different frontier points.
- **Hypothesis:** semantic LM condition/behavior functions may have nonlinear composition or private residual structure; this experiment does not test that.
