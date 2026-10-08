# MA-419 — modulated periodic activation views

Status: **FAIL under the frozen Mirror-specific gate; development screen verified, fresh sealed.** Prior art PA69, Modulated Periodic Activations for Generalizable Local Functional Representations.

## H — Hypothesis
A shared periodic synthesis model addressed by compact latent codes can reconstruct held-out smooth signals with better quality/actual-byte/throughput tradeoffs than a generic latent-concatenation model.

## T — Frozen setup
For each seed, 256 training signals teach a shared model and 64 held-out signal codes are evaluated zero-shot. Targets are four-term 1D periodic functions whose amplitudes, frequency offsets, and phases are smooth functions of an 8D code. Compare periodic activation modulation, generic concatenation MLP, native direct harmonic conditioning, support-adapted private harmonic coefficients, and oracle harmonic state. The native harmonic model is an exact algebraic control. Serialized compressed NPZ bytes include model and inference state. Fresh seeds 41911–41913 remain sealed unless both development seeds pass.

## H/T gate
See `PROTOCOL.json` for fixed updates, seeds, metrics, payload contract, and thresholds. No fresh/audit tuning.

## D — Outcome
The periodic model's held-out NRMSE was **0.000134 / 0.000160**, versus **0.5860 / 0.6202** for generic latent concatenation. Total serialized payload was **2,101 / 2,089 bytes** versus **11,815 / 11,813 bytes** (0.178 / 0.177x); steady-state throughput was **4.93M / 4.83M** versus **4.89M / 4.93M** logical queries/s (1.007 / 0.979x). All three numeric thresholds against concatenation passed.

The native direct-harmonic control used exactly the same output equation and learned state: its outputs and payload sizes matched the Mirror model exactly in both seeds. Therefore the preregistered no-alias condition failed and the Mirror-specific result is **FAIL**. Mirror/native fit wall clock was 4.09/4.16s and 3.79/3.88s; concat was 4.81/4.52s. Private per-signal support adaptation at 400 updates was weak (NRMSE 0.531/0.569), while the serialized oracle was 0.00193/0.00206. Payload hashes and metrics replayed successfully.

## C — Strongest counter-hypothesis
The synthetic teacher itself uses a smooth code-to-harmonic amplitude/frequency/phase map, so the experiment strongly favors a matching harmonic inductive bias over a generic tanh MLP. Because native direct conditioning is algebraically identical, the observed advantage is not evidence for Mirror-specific functionality.

## U — Still unconfirmed
Natural signal tasks, off-family frequency coverage, converged private support adaptation, and fused or optimized serving. The simple arithmetic proxy omits the cost of transcendental sine operations; measured query throughput is therefore the runtime comparison for this screen.

**Evidence labels:** the numbers and exact native alias are facts; FAIL follows from the frozen Mirror-specific gate; natural-domain usefulness is a hypothesis.
