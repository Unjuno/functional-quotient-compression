# MA-327 — Factorized layer x expert Tucker address

Status: FAIL for Mirror-specific byte/quality value. Dedicated branch: `research/ma-327-factorized-layer-expert-tucker-20261008`.

## H — hypothesis

For a planted rank-2 layer x expert coefficient product over a shared two-matrix Tucker bank, a factorized coordinate would recover held-out combinations with test nMSE <= 1e-5 while saving >=20% actual bytes versus flat pair coefficients and >=10% versus the ordinary coefficient-product control.

## T — task and execution

Synthetic regression task: 4 layers x 4 experts, each maps 16-D inputs to 16-D outputs. A shared bank of two 16x16 matrices is addressed by either flat pair coefficients or layer/expert rank-2 products. Four diagonal layer-expert combinations are withheld from coefficient fitting for the product rows. Controls were hard tying, flat pair coefficients, ordinary product coefficients, Mirror product coefficients, and independent full matrices. Each condition received 500 Adam updates. Development seeds were 32701/32702. Fresh seeds were 32711/32712/32713. Fresh quality used models reloaded from serialized FP16 payloads.

The protocol was amended before fresh access to pair identical initialization and minibatch seeds between ordinary-product and Mirror-product rows. This isolates whether their apparent difference is only optimizer noise. The gates and seeds did not change. See `PROTOCOL.json`.

## D — decision

**FAIL for the Mirror-specific claim.** The ordinary coefficient product matched Mirror exactly in all three fresh worlds: same 1,978-byte payload, same SHA-256, and identical test/held-out metrics. Both product forms were 10.6% larger than the flat pair control (1,788 bytes), so neither saved bytes under the actual archive format. The frozen success gate required Mirror to be at least 10% smaller than the ordinary product and <=80% of flat-pair bytes; it failed both storage requirements.

## C — strongest counter-hypothesis

“Factorized layer x expert addressing” here is ordinary rank-2 coefficient factorization. The Mirror label introduces no additional function or compression beyond that native parameterization. Splitting the two factor arrays into separate NPY members also adds archive overhead, making the factorized form larger than the flat code at this small scale.

## U — unresolved

This is a small synthetic linear bank, not a trained MoE or transformer. The equal analytical MAC proxy is not an implementation-level operation count; inference throughput was not benchmarked. Larger dimensions may amortize factor-array/archive overhead. No claim about all Tucker deployments is made.

## Fact / interpretation / hypothesis

**Fact:** Fresh ordinary/Mirror product payloads were 1,978 bytes each, hash-identical within each seed, and metric-identical. All three seeds show +10.6% bytes versus flat coefficients. Product held-out nMSE was 7.02e-5, 1.85e-7, 1.74e-7 in the three worlds, exactly the same for both product rows. Fifteen FP16 payload/hash/metric rows replayed exactly; five tests pass.

**Interpretation:** The tested factorized address is a re-expression of ordinary low-rank coefficient factorization and did not improve the measured storage frontier.

**Hypothesis:** If an alternate Mirror coordinate is useful here, it must add a function or byte-amortization property absent from direct coefficient products; this run provides no evidence for that.

## Results and verification

- `RESULTS_CORE.csv`: all fresh method rows.
- `artifacts/fresh/`: 15 serialized payloads and three metrics records.
- `source/verify.py`: payload byte/hash and FP16-reloaded metric replay.
- `VERIFICATION.json`: verification record.
