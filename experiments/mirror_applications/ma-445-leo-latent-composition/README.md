# MA-445 — LEO latent composition of known skills

## H — Hypothesis

A summed 2D latent code decoded by the structured Mirror map will compose held-out additive skill pairs within 1.10x direct task-vector NRMSE and use fewer actual amortized inference bytes at N=20.

## T — Test

Synthetic 8D regression with six skills whose teacher vectors lie in a fixed 2D subspace. Six pairs held out; 3 fresh worlds × 3 seeds × 6 pairs × 3 replicates, plus four refinement budgets and shared/task-vector/LEO controls. A2 corrected a rank mismatch before the valid run; A1 is excluded. A3 corrected exact serialization accounting without changing trained models or outcomes.

## D — FAIL

Mirror quality is strong (step 0 mean NRMSE 1.0e-5), but N=20 actual payload is 28,645B versus direct task vectors 28,649B, only 4B saved. The practical byte gate fails. LEO is 28,773B.

## C — Strongest counter-hypothesis

This aligned additive family is already solved by ordinary task-vector arithmetic; the tiny byte delta is serializer overhead, not a Mirror-specific benefit.

## U — Unknown

No evidence yet for natural task composition, nonlinear skill families, runtime wins, or additional independent capacity.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `VERIFICATION.json`, and `artifacts/` for frozen protocol, measurements, serialized payloads, hashes, and exact accounting.
