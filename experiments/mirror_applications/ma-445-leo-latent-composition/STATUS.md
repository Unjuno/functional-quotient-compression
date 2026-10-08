# MA-445 status

**FAIL — quality passes for the structured Mirror map, but the actual-byte gate fails.**

## H / T

Tested whether six reusable skill codes compose on held-out pairs through a structured 2D Mirror map. A2 matched the teacher rank to 2D. Three fresh worlds (44530–44532), three seeds, all six held-out pairs, three replicates, four refinement budgets; 2,592 method rows. A1 was invalidated and excluded.

## D — Fact

At step 0, Mirror mean NRMSE was 0.000010 across fresh rows; direct task-vector arithmetic was 0.000043; generic LEO was 0.000117. At N=20, exact serialized shared-bank plus task-address bytes averaged 28,645B for Mirror, 28,649B for direct task vectors, and 28,773B for LEO. Mirror saves only 4B (0.014%) versus task vectors.

## Interpretation

The structured code composes the aligned low-rank skill family accurately, but the storage difference is negligible and does not meet the registered practical byte gate. This does not establish useful generalization beyond an aligned synthetic additive family or independent capacity from pair counts.

## C — Counter-hypothesis

The skill family is exactly additive in a shared 2D subspace, so ordinary task-vector addition already captures the function. Mirror's few-byte difference is serializer metadata noise, not a meaningful compression gain.

## U — Unconfirmed

Natural/nonlinear skill composition, runtime advantage, and robustness to out-of-subspace skills remain untested.

A0/A1 exploratory data are excluded. See `PROTOCOL.json` amendments and `artifacts/serialized_accounting.csv` for exact byte records.
