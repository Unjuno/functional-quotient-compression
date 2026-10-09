# MA-397 status

- Status: **FAIL** at frozen vector/collision-quality gates; fresh remains sealed.
- Branch: `research/ma-397-product-address-mirror-vocabulary-20261009`
- Base commit: `9caa7a01`
- Prior art: PA43, PA58 and PA59.
- Development seeds: 39701, 39702.
- Fresh seeds: 39711, 39712, 39713 (sealed).
- Address audit: 1,024 addresses, each occupied by exactly two tokens.
- Serialized payloads: eight; sizes, SHA-256 and metrics replay exactly.
- Tests: four passed.

## Result

Mirror used 13,941/13,964B (11.25% of full, 64.9% of native coefficients) but had embedding NRMSE .131/.136 and pair-separation NRMSE .128/.128, while direct coefficients reached ~.008/.007 and pair separation ~.008/.007. Pure addition separated no occupants (pair error 1.0). Fresh remained sealed. See README and RESULTS_CORE.csv.
