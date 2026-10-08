# MA-419 — modulated periodic activation views

Status: SCREENING; frozen protocol before runs. Prior art PA69, Modulated Periodic Activations for Generalizable Local Functional Representations.

## H — Hypothesis
A shared periodic synthesis model addressed by compact latent codes can reconstruct held-out smooth signals with better quality/actual-byte/throughput tradeoffs than a generic latent-concatenation model.

## T — Frozen setup
For each seed, 256 training signals teach a shared model and 64 held-out signal codes are evaluated zero-shot. Targets are four-term 1D periodic functions whose amplitudes, frequency offsets, and phases are smooth functions of an 8D code. Compare periodic activation modulation, generic concatenation MLP, native direct harmonic conditioning, support-adapted private harmonic coefficients, and oracle harmonic state. The native harmonic model is an exact algebraic control. Serialized compressed NPZ bytes include model and inference state. Fresh seeds 41911–41913 remain sealed unless both development seeds pass.

## H/T gate
See `PROTOCOL.json` for fixed updates, seeds, metrics, payload contract, and thresholds. No fresh/audit tuning.
