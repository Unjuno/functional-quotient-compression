# MA-611 — Neural Interpreter Mirror function codes

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `5fd9a420` (worker-ready baseline plus MA-602–610 evidence)
Prior art: PA128 Neural Interpreters

## Hypothesis

H: For a shared function interpreter with a two-channel phase basis and discrete frequency signature, one Givens angle per logical function preserves support-adapted and composed held-out function quality while using fewer serialized code bytes than an unconstrained two-value function code.

## Mirror insertion

> **Mirror insertion:** this experiment adds one angle `m` to a shared sine/cosine function basis selected by a frequency signature, so each logical function is a phase view executed by the same interpreter instead of storing two independent basis coefficients.

This is a deterministic mechanism screen. The executor is analytic and shared; only function codes and signatures are serialized. It is not a trained neural interpreter result.

## Gates

PASS requires Mirror and ordinary coefficient-code query NRMSE2 within 1e-4 in both seeded worlds, composition NRMSE2 <=1e-5 for both, and lower actual serialized code-bank bytes. Otherwise FAIL. Report code-adaptation work and inference compute separately.

## H / T / D / C / U

**H — hypothesis:** one Mirror angle per logical phase function matches a two-value unconstrained function code on support-adapted queries and phase composition, while reducing serialized code-bank bytes.

**T — execution:** deterministic analytic shared sine/cosine interpreter with four discrete frequency signatures; 64 phase functions (16 per signature), 32 support samples and 256 query samples/function; 32 same-signature composed functions. Two seeds 61101/61102. Mirror codes use a 720-angle support grid; ordinary coefficients use least squares. No optimizer updates or fresh set. Serialized both original and composed code banks.

**D — FAIL under the frozen gate:** Mirror payload is 2,845 B versus 3,229 B coefficients (11.9% smaller), but composed-function NRMSE2 is 1.80e-5/1.42e-5, above the 1e-5 gate. Ordinary coefficient codes are near machine-zero on query and composition. Mirror query NRMSE2 is 7.2–7.5e-6, within 1e-4 of the coefficient control, but uses a 720-point search per task (1,474,560 candidate/sample operations versus 2,048 support examples for least squares), twice the query compute proxy, and 1.3–1.4 ms versus 0.91 ms CPU latency per 16,384 queries. This misses the strict composition gate; no fresh data apply.

**C — strongest counter-hypothesis:** the apparent composition miss comes from quantizing the support-fit angle to the fixed 720-point grid; a continuous optimizer could reduce it, but the native two-value code solves the support least-squares problem directly and is both more accurate and faster.

**U — boundaries:** this is an analytic Fourier function family rather than a trained Neural Interpreter; frequency signatures are fixed and there are no learned executor weights. It does not establish broad modular program generalization.

## Facts / interpretation / hypothesis

- **Fact:** 4/4 saved payloads match size/hash and replay exactly; two tests pass.
- **Interpretation:** a one-angle code can nearly match coefficient codes and saves about 12% on this small serialized bank, but the chosen adaptation algorithm costs substantially more and its quantization misses the composition threshold.
- **Hypothesis:** a learned or closed-form continuous Mirror-code inference rule may retain the code-size advantage without the grid-search cost; that would require an amended experiment with its own frozen protocol.
