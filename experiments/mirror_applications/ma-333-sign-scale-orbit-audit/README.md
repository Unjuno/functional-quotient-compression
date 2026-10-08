# MA-333 — Mirror sign/scale orbit audit

Status: FAIL as additional functional capacity; symmetry audits passed. Dedicated branch: `research/ma-333-sign-scale-orbit-audit-20261008`.

## H — hypothesis

ReLU positive rescaling and tanh hidden sign flips, when coupled across incoming and outgoing tensors, change parameter coordinates while preserving functions. Such addresses create no additional logical functions.

## T — execution

Trained 8-12-3 single-hidden-layer MLPs for 300 Adam updates on synthetic regression tasks. For each world, audited (a) a positive ReLU scale with reciprocal outgoing scale and (b) a tanh sign flip applied to both incoming and outgoing weights. Negative controls changed only incoming weights/biases. Development seeds 33301/33302; fresh seeds 33311/33312/33313.

FP32 algebraic symmetry error was evaluated separately from FP16 base/view quantization error after actual serialized payload reload. The protocol was amended before fresh access to make this distinction; seeds and training settings did not change.

## D — decision

**FAIL as additional functional capacity.** ReLU scale and tanh sign symmetry preserve the FP32 function, but do not create another function. Fresh FP32 max output differences were <=2.98e-6 for ReLU scaling and exactly 0 for tanh sign flips. FP16 quantization is separate: the loaded base/view discrepancy reached 0.0108 for ReLU and was zero for the sign-only tanh transform. The uncoupled negative controls changed outputs strongly.

Each base payload was 1,408B; transformed payloads with paid 24B scale/sign code and archive overhead were 1,652B. These codes add storage for coordinate variants while retaining the same function.

## C — strongest counter-hypothesis

The observed FP16 difference under ReLU scaling is caused by independent rounding of transformed weights, not new predictive behavior. A distinct function would require a change that survives function-space comparison after accounting for quantization.

## U — unresolved

Only one-hidden-layer synthetic MLPs were tested. Other activations, architectures, learned gauge-aware codes, and sign/scale families outside the exact identities remain open.

## Fact / interpretation / hypothesis

**Fact:** Five seeds per audit condition ran. FP32 symmetry max difference was <=2.98e-6 (ReLU) and 0 (tanh). FP16 loaded discrepancy reached 0.0108 (ReLU). Twenty symmetry payload/hash/metric rows replayed; four tests pass.

**Interpretation:** Positive scale and sign coordinates are gauge freedom for these networks. Parameter orbit size must not be counted as logical capacity.

**Hypothesis:** Future symmetry-derived Mirror candidates need an explicit function-change test before any multiplicity claim.

## Family ruling

MA-332 permutation and MA-333 sign/scale audits are consecutive P0 results with the same structural outcome: exact symmetry Views add zero function multiplicity. Pause pure permutation/sign/scale orbit proposals as capacity mechanisms pending a function-changing extension. MA-335 group-action experts is a separate insertion and remains next.

## Evidence files

- `RESULTS_CORE.csv`: fresh rows.
- `artifacts/`: actual FP16 inference payloads and metric records.
- `source/verify.py`: payload/hash and FP32 symmetry replay.
- `VERIFICATION.json`: verification record.
