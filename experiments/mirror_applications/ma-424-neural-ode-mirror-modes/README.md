# MA-424 — Neural ODE vector-field modes

Status: **FAIL at the frozen development screen; fresh seeds sealed.** Prior art PA70 (Neural ODEs) and PA71 (Deep Equilibrium Models).

## H — Hypothesis
A shared neural vector field with a small state-coordinate View can realize multiple continuous dynamics with lower actual bytes than independent mode-specific fields while retaining solver quality and throughput.

## T — Frozen setup
Use a fixed contractive 8-32-8 tanh vector field and sixteen common-angle, four-pair Givens views. Integrate 32 shared initial states per mode with 64 Euler steps; compare against a 256-step RK4 reference. Compare coordinate-view inference, a native generated-weight algebraic control, independent full transformed networks, and one unconditioned shared field. No model training is performed. Payload bytes include all stored weights, mode codes/IDs, and metadata. Fresh worlds 42411–42413 are sealed.

See `PROTOCOL.json` for frozen seeds, gates, storage accounting, and runtime protocol.

## D — Outcome
The 64-step Euler trajectory relative RMSE against 256-step RK4 was **3.82e-6 / 3.90e-6** for Mirror and **3.84e-6 / 3.90e-6** for independently stored transformed networks. Actual payload was **2,566 / 2,562 bytes** for Mirror versus **16,557 / 16,612 bytes** for independent fields (0.155 / 0.154x). Thus the aligned representation has a large byte reduction against full copies.

The native generated-weight control used **byte-identical payloads and trajectories**. Mirror throughput was **2.54M / 4.41M** vector-field evaluations/s, versus **5.36M / 4.96M** for that native control (0.474 / 0.889x); seed 1 missed the 0.80 runtime threshold. The exact native alias also triggered the frozen FAIL condition. Mode diversity RMS was **7.39e-4 / 5.55e-4**, small in this deliberately weak contractive field. The one shared unconditioned field had relative trajectory error 7.03e-4 / 5.87e-4. No training updates occurred.

## Execution correction and provenance
The first execution retained autograd graphs during inference timing. A second no-grad execution also did unnecessary native weight materialization in the Mirror load path. Both initial result sets and payloads remain under `runs/initial_*` with notes; the canonical results use no-grad inference and method-specific materialization. Seeds, solver steps, payloads and thresholds were unchanged. Canonical metric and payload replay passed.

## C — Strongest counter-hypothesis
Coordinate conjugacy folds into the native MLP's input and output weights, so its function and paid state are ordinary generated-weight conditioning. The weak vector field also produces very small differences across views, limiting the usefulness of the demonstrated multiplicity.

## U — Still unconfirmed
Learned Neural ODEs, application-relevant trajectory diversity, adaptive solvers, stiffness/stability under wider modes, and fused/GPU throughput. This is not a learned-model capacity result.

**Evidence labels:** measured errors, bytes, runtime and exact native alias are facts; FAIL follows from the frozen native-alias/runtime gate; useful nonlinear state-dependent Views remain a hypothesis.
