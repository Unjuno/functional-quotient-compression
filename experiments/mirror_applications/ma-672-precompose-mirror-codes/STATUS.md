# MA-672 status

- Status: FAIL (Mirror-specific claim); generic precomposition frontier is promising on this synthetic screen
- Branch: `research/ma-672-precompose-mirror-codes-20261104`
- Development: seeds 67201/67202; three implementation amendments preserve earlier attempts
- Fresh/audit: sealed after exact native-control alias

## H — Hypothesis

Precomposing input-gated LoRA expert coefficients over a rank-4 shared factor bank preserves output with fewer bytes and active MACs than evaluating all independent LoRAs; Mirror must beat native coefficient precomposition.

## T — Treatment and controls

Synthetic D=32, E=16, R=4, 512 examples; residual RMS 0/.05; two seeds. Compared exact independent LoRA A/B factors, selected top-2 independent experts, shared rank-4 factor precomposition, native coefficient precomposition, and paired rank-2 private residual controls at rho=.05. Initial dense-atom, unpruned-top2, and dense-full-bank attempts are preserved under `source/attempts/`; each correction was protocol-amended before rerun. Tests replay metrics from serialized payloads.

## D — Decision

FAIL for Mirror-specific value. At rho=0, shared precomposition is exact at ~8.0KB versus ~21.9KB full independent rank-4 LoRAs, and 950,272 versus 2,883,584 MAC proxy. At rho=.05, shared-only NRMSE is .010–.012 versus exact full-rank independent LoRAs; rank-2 residual slightly improves error to .009–.011 but doubles shared payload to ~16.2KB and MAC proxy to ~2.0M. Native coefficient controls exactly match outputs and compute; their payloads are 14B larger due to key naming, not a functional difference. Top-2 independent routing has .24–.31 NRMSE. Fresh stayed sealed.

## C — Strongest counter-hypothesis

Linearity lets ordinary low-rank coefficient aggregation perform the same function and compute. The gain over separate expert factors is standard shared-basis factorization and precomposition, not Mirror-specific freedom.

## U — Unconfirmed

Natural trained adapter quality, learned gates, optimized GPU kernels, and hardware speedup remain untested. MAC proxy is not latency; measured CPU timings are only a tiny synthetic screen.

## Evidence labels

- Fact: serialized payload-replayed errors, actual bytes, MAC proxies, and wall times are in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: sharing and precomposing rank-4 aligned LoRAs improves the tested storage/compute frontier over independent rank-4 experts; rank-2 private state was unnecessary to pass the .02 output gate at rho=.05.
- Hypothesis: Mirror-specific gains require a composition mechanism beyond native linear coefficient aggregation.
