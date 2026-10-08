# MA-393 report

## H — Hypothesis
At equal or lower actual bytes than native rare-token dimension expansion, a per-token Givens View over a 4D rare-token basis can improve rare-token task quality.

## T — Trial
Synthetic vocabulary of 128 with 32 common and 96 rare tokens. Common tokens received 8× sampling probability. The teacher generated rare vectors from two 2D latent components, a token angle, and a shared projection. Five methods (full 16D table, adaptive 4D, adaptive 5D equal-increment control, Mirror angle, scalar gate) ran for 256 Adam updates on 32,768 examples in two development worlds. The class labels and examples were shared across methods. Actual deterministic ZIP/NPY FP16 payloads were reloaded for evaluation.

## D — Decision
**FAIL.** Fresh seeds 39311–39313 remain sealed.

| Seed | Full rare acc. | Adaptive4 rare acc. | Mirror rare acc. | Mirror bytes / adaptive5 | Mirror throughput / best adaptive native |
|---:|---:|---:|---:|---:|---:|
| 39301 | 1.000 | 1.000 | 1.000 | 3964 / 3774B | 0.858×
| 39302 | 1.000 | 0.990 | 0.938 | 3964 / 3774B | 0.862×

The Mirror did not beat adaptive4 rare accuracy by the required 5 points and missed full-table rare quality by 6.2 points in one world. Its payload was 1.050× adaptive5, above the 0.90× gate, and 0.750× full16, above the 0.60× gate.

## C — Strongest counter-hypothesis
The task was too easy to expose rare-capacity pressure: every baseline saturated near perfect accuracy. Mirror paid for per-token angles and trigonometric operations without a measured quality gain.

## U — Unconfirmed
No real language-model token frequencies, natural tail distribution, fresh-world replication, near-convergence capacity result, or optimized fused runtime is established.

## Evidence classes
**Facts:** result values, payload bytes/hashes, timing, and exact replay are in `RESULTS_CORE.csv`, `runs/`, and `VERIFICATION.json`.

**Interpretation:** this frozen screen fails the rare-quality, byte, and runtime gates.

**Hypothesis:** a more severe tail distribution could reveal a capacity regime where learned Views help, but changing that task now requires a new protocol.
