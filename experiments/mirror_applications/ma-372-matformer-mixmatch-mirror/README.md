# MA-372 — MatFormer Mix'n'Match factorized Mirror codes

Status: **PROTOCOL FROZEN BEFORE DEVELOPMENT**

## H — falsifiable hypothesis

Layer×width factor codes learned only from homogeneous-width subnetworks can compose into useful unseen mixed-width subnetworks, improving on the native nested model and equal-shape FiLM while costing substantially less than a code per mixed configuration.

## Prior art delta

PA53/PA107 establish nested FFN models and Mix'n'Match extraction. This experiment isolates whether factorized Mirror codes improve quality on held-out mixed-width layer assignments. It does not count the 27 configurations as 27 independent functions.

## T — frozen protocol

Three-layer nested-width MLP (64 maximum, widths 16/32/64), digits v1.8.0, per-world stratified 60/20/20. Train a sandwich-sampled supernet with distillation; fit each layer×width Givens/FiLM factor only on homogeneous-width subnet tasks. Evaluate all 24 non-homogeneous ordered triples, which are excluded from code fitting. Dev worlds 37200/37201 choose LR from {0.003,0.01}; fresh 37210/37211/37212 are sealed. Compare native nested, factorized Mirror, equal-shape FiLM, and direct per-configuration code bank. Complete actual serialized bytes are authoritative.

Gates and exact budgets are frozen in `PROTOCOL.json`.


## Results

Fresh mean accuracy across mixed configs: nested 94.41%, factorized Mirror 94.56%, equal-byte FiLM 94.70%, direct per-config Mirror 94.70%. Factorized Mirror uses 57,968B (77.5% of the direct bank's 74,803B), above the frozen 60% gate. **Decision: FAIL** for the registered Mirror-specific quality/storage claim. `STATUS.md` gives H/T/D/C/U; `VERIFICATION.json` records replay checks.
