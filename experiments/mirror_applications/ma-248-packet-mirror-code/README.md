# MA-248 — PTP random variable represented as packet Mirror code

Status: **FAIL (Mirror-specific frontier not established)**  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `f91625f2fe1b110593b16605c26fc9c7675c1824`

## H — falsifiable hypothesis

A single packet address, read through phase-specific low-description Mirror views, can generate a jointly consistent P=4 trajectory when branch uncertainty is shared across phases; when branch entropy is independent per phase, the required address entropy grows from one to four bits. Any quality/storage gain must beat a simple shared-code control, not just exploit the smaller correlated source.

## T — execution

PA10 (Parallel Token Prediction, ICLR 2026) feeds one random auxiliary per future position. Each future token is a deterministic function of context and its own plus preceding auxiliaries. TM001 found that factorized period slots failed when one hidden branch was shared across a packet; a small packet latent helped but did not close the gap.

This experiment used a tiny causal two-block phase-slot decoder over a synthetic P=4 state-transition task. There were 16 states, 8 rules and two random branch-specific permutation tables. In the correlated condition one binary branch was sampled once per packet (one source bit). In the independent condition each phase sampled its own branch (four source bits). Methods were PTP-style per-slot auxiliaries, no-aux direct slots, broadcast shared packet code, scalar-gated shared code, shared code with phase-specific Givens views, and untied per-phase embeddings. All methods saw the same minibatch stream per world/mode/rate in final dev/fresh runs.

The first development pass used one causal block and had low joint accuracy. A pre-fresh amendment used two blocks so later slots could attend to earlier contextualized slot representations. A second pre-fresh audit found method-specific minibatch streams; a further development-only amendment matched these streams across controls. Both prior development runs are preserved and tagged in `RESULTS_CORE.csv`. The final matched-minibatch v3 development world 24800 selected LR 0.01 across both source modes and six methods. Fresh worlds 24801–24803 used that locked rate and source.

## D — FAIL for the registered Mirror-specific claim

| Correlated source, fresh world | PTP joint packet accuracy | Broadcast shared code | Scalar gate | Mirror views | Untied |
|---|---:|---:|---:|---:|---:|
| 24801 | 100.0% | 100.0% | 84.2% | 26.5% | 100.0% |
| 24802 | 88.9% | 100.0% | 57.9% | 100.0% | 70.4% |
| 24803 | 95.8% | 100.0% | 42.2% | 99.3% | 86.7% |

Mirror met the >=99% gate in 2/3 correlated fresh worlds, but failed badly in 24801. The byte-near broadcast control matched the shared one-bit source and achieved 100% in all three fresh worlds. It used 31,840 serialized model bytes versus 32,218 for Mirror (378 bytes fewer). Mirror also used the same one source bit as PTP in the correlated condition; there was no random-address saving. The scalar-gate payload was 32,223 bytes, but it was unstable and did not match the broadcast result. Thus the registered Mirror-specific storage/quality claim failed.

Under independent per-phase branch entropy, all methods had low exact-packet accuracy. Across the three fresh worlds, PTP joint accuracy was 0.95%, 2.10%, 2.25%; Mirror was 1.17%, 1.93%, 1.95%; and untied was 1.90%, 3.22%, 3.98%. This does not establish an entropy limit: the PTP and untied controls also struggled under this fixed-width, fixed-update task.

### Storage and compute

In correlated mode, exact serialized model payloads were: PTP 31,834 B; broadcast 31,840 B; Mirror 32,218 B; scalar gate 32,223 B. Each consumed one source-address bit per packet. In independent mode PTP/Mirror consumed four bits; payloads were 31,835 B / 33,115 B. These payloads include serialized state dict and deterministic config metadata.

For correlated fresh runs, median training wall time per method was PTP 7.65 s, broadcast 7.22 s, scalar gate 7.50 s, Mirror 8.35 s. Median one-thread CPU inference throughput was PTP 57.5k examples/s, broadcast 47.4k, scalar gate 52.9k, Mirror 43.0k. The MAC proxy per run was 3.465B for broadcast/scalar-gate, 3.458B for Mirror, and 3.465B for PTP. The tiny eager CPU benchmark is noisy and does not support a general throughput claim.

## C — strongest counter-hypothesis

The Mirror collapse in world 24801 may be an optimization/initialization failure rather than a representational limit: two other correlated fresh worlds reached >=99%, and the untied upper control itself fell to 70.4% in world 24802. Still, broadcast achieved 100% in all worlds with fewer bytes, so this counter-hypothesis cannot rescue a Mirror-specific advantage in this run.

## U — not established

Natural-language PTP distillation, longer packets, more realistic continuous PTP auxiliaries, fixed-byte near-convergence capacity, stronger optimization/init, optimized kernels, and the independent-mode entropy boundary remain untested. The result is limited to this synthetic transition process and fixed 1,200-update budget.

## Fact / interpretation / hypothesis

- **Fact:** Mirror passed the correlated exact-packet threshold in 2/3 fresh worlds and failed in one. Broadcast shared code passed in 3/3 and used 378 fewer model bytes. Address entropy was equal between Mirror and PTP in each source mode. The independent mode had low exact accuracy for every method.
- **Interpretation:** The packet code can express a correlated shared branch, but the Givens view did not yield a reliable quality/storage frontier improvement over ordinary shared-code broadcast. Fixed-budget independent mode was too weak across controls to infer a fundamental boundary.
- **Hypothesis:** The correlated seed variance likely reflects optimization sensitivity; a stronger initialization or longer training may remove it, but would require a new preregistered amendment/experiment.

## Reproduction and verification

- Tests: `python -m pytest -q experiments/mirror_applications/ma-248-packet-mirror-code/tests` (4 passed).
- Frozen protocol/source hashes and the pre-fresh stage are in `FREEZE_MANIFEST.json`.
- All 36 fresh rows were re-trained and replayed. Maximum NLL difference was 4.62e-10; maximum joint/accuracy/path difference was 5e-9; every serialized payload byte count matched exactly. See `VERIFICATION_REPLAY.json`.
