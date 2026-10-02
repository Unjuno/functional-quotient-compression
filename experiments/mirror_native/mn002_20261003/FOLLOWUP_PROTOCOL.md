# MN002R: fresh-table follow-up

Designed only after the MN002 audit. Mirror4 beat Dense36 in all 3 MN002 seeds; Mirror16 lost to Dense48 in all 3. This follow-up tests the promising small S4 configuration against both Dense36 and DirectGate24 on a new random permutation table (161803), new sequence streams and 3 fresh training seeds (211/212/213). All other training settings stay fixed. All 9 final-step checkpoints must be locked before audit. No result-based early stopping or task/hyperparameter search is allowed.

Report all 9 runs even if the effect vanishes or reverses. This is a second finite task table, not natural language, a universal capacity proof or a statistically powered population study. DirectGate24 uses 24 more scalars (+0.136%) but the same linear MACs as Mirror4/Dense36. If a simple gate performs similarly, do not claim that a bank of discrete mirror states is uniquely necessary.

## Erratum to the frozen machine-readable protocol

The prose `audit_policy` in `followup_protocol.json` inherited the words “all 24” from the main protocol. The operative variant/seed arrays specify 3 × 3 = 9 runs, and `FROZEN_FOLLOWUP.json` locks all 9. The descriptive count should read “all 9”. The frozen JSON is preserved byte-for-byte so checkpoint provenance hashes remain verifiable; no data, training, selection, or audit order was changed.
