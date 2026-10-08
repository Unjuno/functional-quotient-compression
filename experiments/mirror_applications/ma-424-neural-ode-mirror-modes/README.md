# MA-424 — Neural ODE vector-field modes

Status: SCREENING; frozen protocol before runs. Prior art PA70 (Neural ODEs) and PA71 (Deep Equilibrium Models).

## H — Hypothesis
A shared neural vector field with a small state-coordinate View can realize multiple continuous dynamics with lower actual bytes than independent mode-specific fields while retaining solver quality and throughput.

## T — Frozen setup
Use a fixed contractive 8-32-8 tanh vector field and sixteen common-angle, four-pair Givens views. Integrate 32 shared initial states per mode with 64 Euler steps; compare against a 256-step RK4 reference. Compare coordinate-view inference, a native generated-weight algebraic control, independent full transformed networks, and one unconditioned shared field. No model training is performed. Payload bytes include all stored weights, mode codes/IDs, and metadata. Fresh worlds 42411–42413 are sealed.

See `PROTOCOL.json` for frozen seeds, gates, storage accounting, and runtime protocol.
