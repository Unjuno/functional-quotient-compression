# MA-612 — Factorized function signature and execution View codes

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / ROUTING
Base commit: `4532f4b8` (worker-ready baseline plus MA-602–611 evidence)
Prior art: PA128 Neural Interpreters

## Hypothesis

H: Separating a discrete function signature (frequency family) from a reusable execution View (phase) lets a shared interpreter realize held-out signature/code pairings with lower actual bytes than per-function codes; a Givens phase code should further reduce code bytes versus ordinary two-coefficient phase codes without losing function quality.

## Mirror insertion

> **Mirror insertion:** this experiment factors the interpreter address into a discrete signature `s` selecting a shared frequency basis and an execution coordinate `m` rotating its sine/cosine channels, so each signature can pair with reusable logical phase functions.

## Scope and gate

Deterministic analytic interpreter screen, not a learned Neural Interpreter. Evaluate all 64 pairings of four signatures and sixteen phase Views, marking 16 pairings held out from the explicit pair table. PASS requires exact routing and query MSE <=1e-6 on held-out pairings, plus Mirror actual payload smaller than both an unfactorized function-code table and a factorized ordinary coefficient-code bank.

## H / T / D / C / U

**H — hypothesis:** splitting a function signature from a reusable execution View supports held-out cross-pairings, and one Givens phase per View further reduces serialized code bytes versus both a per-function table and ordinary factorized coefficients.

**T — execution:** deterministic analytic sine/cosine interpreter; four frequency signatures × sixteen phase codes = 64 functions, with 16 pairings marked held out from an explicit pair table. Compared per-function coefficient vectors, factorized signature + coefficient-code tables, and factorized signature + Mirror angles. Two seed labels 61201/61202 produce the same exhaustive phase/signature grid; they test serialization/replay determinism, not independent data replications. No optimizer updates. Measured actual `torch.save` bytes, held-out routing/output, arithmetic/trigonometric operation proxies and CPU query latency.

**D — PROMISING for this scoped deterministic screen:** all representations route 100% correctly and have held-out NRMSE2 below 2.4e-14; max output error is <=5.4e-7. Mirror payload is 2,401 B versus 2,465 B factorized coefficient codes (2.6% fewer) and 2,533 B per-function codes (5.2% fewer). It uses 2 arithmetic ops + 1 trig op/query versus 3 + 2 for coefficient codes. Measured 16,384-query latency was 0.77–0.89 ms Mirror, 1.32 ms coefficient codes, and 1.14–1.19 ms per-function codes. No fresh set applies to this exhaustive algebra screen.

**C — strongest counter-hypothesis:** byte savings over the already factorized ordinary-code control are only 64 B in this payload, and the repeated seeds do not add independent evidence. This demonstrates representation arithmetic, not a learned router's ability to discover signatures or execution Views.

**U — boundaries:** analytic Fourier functions with predeclared signatures and Views; no learned interpreter, task discovery, routing noise, or natural-language behaviors.

## Facts / interpretation / hypothesis

- **Fact:** all six payloads match exact file size and SHA-256; all 64 pairings have exact routes; two tests pass.
- **Interpretation:** separating reusable routing signatures and execution coordinates exposes a compact product structure; Givens phase codes add a small byte reduction beyond factorization alone.
- **Hypothesis:** the same factorization may help learned function banks only when execution codes truly transfer across signatures; a trained Neural Interpreter experiment must test whether that decomposition emerges from data.
