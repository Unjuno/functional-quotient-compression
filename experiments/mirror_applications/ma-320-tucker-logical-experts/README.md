# MA-320 — Tucker logical experts with Mirror phase addresses

Status: **FAIL** on the frozen 10% storage promotion gate. Dedicated branch: `research/ma-320-tucker-logical-experts-20261008`.

## H — hypothesis

A small phase address per expert over a shared Tucker bank would encode a 48-expert harmonic orbit with at least 10% fewer total bytes than rank-16 free coefficients while preserving each expert's held-out function nMSE <=1e-4. Sixteen off-orbit experts would reveal where private parameters are needed.

## T — what was run

**Fact:** A synthetic post-fit MoE bank had 64 top-1 linear experts mapping 32D inputs to 16D outputs. Forty-eight experts shared an eight-harmonic phase orbit; 16 unrelated experts were generated outside the shared matrix bank. Every expert received 96 support, 64 validation and 128 test inputs. There were no optimizer updates; task ID was oracle-supplied and charged. Development seeds 32001/32002; fresh seeds 32011/32012/32013.

Controls: hard tie, free Tucker coefficients at ranks 1/2/4/8/16, rank-16 Tucker with the same validation-selected private full-matrix fallback, and independent FP16 weights. Mirror used the shared 16-matrix bank, eight shared harmonic amplitudes and one phase per aligned expert; unrelated tasks could use the same private fallback. Every payload charges shared mean/basis, codes, private matrices, task IDs, metadata and deterministic ZIP/NPY headers. See `PROTOCOL.json`.

## D — decision and facts

**FAIL** against the preregistered promotion gate. Across all three fresh seeds, Mirror used 35,886B versus 37,074B for rank-16 Tucker plus the same private fallback: 3.20% fewer bytes, short of the required 10%. Mirror mean test nMSE was 3.88e-6–5.44e-6 and maximum task nMSE was 1.91e-5–2.91e-5, so every task stayed under 1e-4. The direct Tucker fallback control had mean nMSE about 4.6e-8. Both methods allocated 16 private matrices for the 16 unrelated tasks; Mirror encoded the 48 aligned experts with phases.

| Method | Payload | Mean test nMSE | Private experts |
|---|---:|---:|---:|
| Hard tie | 1,558B | 0.101–0.106 | 0 |
| Tucker rank 8 | 11,986B | 0.034–0.036 | 0 |
| Tucker rank 16 | 21,202B | 0.014–0.016 | 0 |
| Tucker rank 16 + private fallback | 37,074B | 4.58e-8–5.25e-8 | 16 |
| Mirror phase + private fallback | 35,886B | 3.88e-6–5.44e-6 | 16 |
| Independent full FP16 | 66,076B | 4.33e-8–4.35e-8 | 0 |

Mirror saved 1,188B against the strongest matched control. Its fit-compute proxy was 1.61B versus 50.3M for Tucker coefficients (~32x); fit wall time averaged 0.35s vs 0.046s. The measured top-1 path computes a matrix-bank view per request; no router or cached materialization is included. CPU requests/s is diagnostic and did not show a consistent Mirror speedup.

## Interpretation, counter-hypothesis and limits

**Interpretation:** One phase can encode many deliberately aligned expert functions and the private fallback cleanly identifies the off-orbit boundary. However, the unrelated experts' full matrices dominate the bank, leaving only a 3.2% total-byte gain over free Tucker coefficients. Phase search adds substantial fitting work. The orbit was planted by construction, so 48 task codes are not 48 independent-capacity evidence.

**C — strongest counter-hypothesis:** the shared Fourier generator is only useful when expert functions already follow that precise orbit; a direct Tucker address is cheaper to fit and has lower error. A larger all-aligned bank might amortize the savings, but is untested and cannot alter this result.

**U:** learned routers, trained neural experts, natural task distributions, larger all-aligned banks, and near-converged task learning remain untested. No capacity claim.

## Verification

Five tests passed. The verifier reproduced all 45 method/seed summaries and 2,240 allocation events exactly. All serialized package hashes matched after reload; max metric difference was zero. Fresh split seeds were not used in development or selection.
