# MA-304 — Supermask plus Mirror transform inside active edges

Status: protocol frozen before development. Dedicated branch: `research/ma-304-supermask-active-mirror-transform-20261008`.

## H — hypothesis

A single phase on two shared sparse edge bases will recover aligned functions without increasing the active edge count, and save at least 10% actual payload over the direct two-coefficient control. An unrelated-task bank tests the private-state boundary.

## Mirror insertion and controls

The base is a 32x32 linear map with exactly 256 active edges per task. Mirror interpolates two shared sparse bases using one task phase. Controls include dense shared weights, fixed sparse tying, a scalar gate, direct two-coefficient shared bases with identical private fallback, SupSup/Piggyback-style binary task masks over one dense backbone, and independent full matrices.

## Frozen protocol

See `PROTOCOL.json`. 128 aligned tasks and 16 unrelated tasks; 64 support, 32 validation and 64 test vectors per task; development seeds 30401/30402 and fresh seeds 30411/30412/30413. No optimizer updates. The actual deterministic ZIP/NPY payload includes all paid bases, indices, task codes, masks, private state and metadata.

## Development observations

Across both development seeds, Mirror used 21,288B versus 21,318B for direct coefficients (30B / 0.14% smaller), with 16 unrelated tasks privately stored in both. Mirror aligned max nMSE was 5.07e-5–5.26e-5; direct was about 1.5e-7. Mirror fit operation proxy was 452,984,832 vs 150,994,944 for direct (~3x). The frozen 10% fresh storage gate is therefore unlikely to pass, but fresh evaluation proceeds without tuning or changing gates.

## H/T/D/C/U report

**H — Hypothesis.** At a fixed 256 active edges per task, a Mirror phase over two shared sparse bases would match aligned tasks at max test nMSE<=0.02 and save >=10% actual payload versus direct coefficients, without extra private fallbacks. Unrelated tasks would show the private-state boundary.

**T — Trial.** 32x32 linear maps; 128 planted phase-orbit tasks plus 16 unrelated sparse maps; 64 support, 32 validation and 64 test vectors per task; two dev seeds and three fresh seeds. Controls were dense shared, a SupSup/Piggyback-style binary-mask heuristic on a shared dense matrix, hard sparse tying, scalar gate, direct two-coefficient shared basis with the same private fallback, phase Mirror, and independent FP16 matrices. Zero optimizer updates. Fresh source was frozen at b953750.

**D — FAIL for the strict storage promotion gate.** Across all three fresh seeds, Mirror payload was 21,288B versus 21,318B direct coefficients (0.141% smaller, below the 10% requirement). It used the same 16 private fallbacks as direct. Aligned max nMSE was 4.28e-5–4.78e-5 for Mirror and 1.14e-7–1.70e-7 for direct; both met the <=0.02 quality threshold. Unrelated task max nMSE was <=5.11e-8 due to private storage. Active edge count was 256 for both methods. Mirror fit proxy was 452,984,832 vs 150,994,944 direct (~3x); eager CPU throughput was 0.47–0.79x direct. Mirror was 72.4% smaller than the binary-mask heuristic (77,008B), but that heuristic's aligned max nMSE was 1.63–1.80, so this is not a matched-quality advantage. Independent full matrices used 295,912B and near-zero error.

**C — Strongest counter-hypothesis.** One scalar phase saves only 30 serialized bytes over two FP16 coefficients because shared bases, private task state and package metadata dominate. Direct coefficients also fit with far fewer operations and lower CPU latency. The apparent mask-only compression trades away most quality.

**U — Unverified.** Synthetic post-fit linear maps only; planted aligned phase orbit, oracle task IDs, no optimizer or learned router; the binary-mask control is a simple heuristic, not a full SupSup reproduction. No trained-model capacity claim or GPU runtime result.

### Evidence categories

- **Fact:** 21 fresh payloads were byte/hash checked and reloaded; 21 summary rows and 3,024 task allocation records replayed exactly. Four tests pass.
- **Interpretation:** A fixed-edge Mirror view recovers useful aligned function diversity without increasing active compute, but does not improve the total-byte frontier over direct coefficients and costs more fit compute and eager CPU time.
- **Hypothesis:** A lower-cost phase solver or much larger aligned bank could amortize shared state; this would still need to beat direct coefficient coding at matched quality in a trained setting.
