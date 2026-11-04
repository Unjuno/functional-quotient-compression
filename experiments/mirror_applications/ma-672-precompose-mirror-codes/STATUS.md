# MA-672 status

- Status: FAIL (Mirror-specific claim)
- Branch: `research/ma-672-precompose-mirror-codes-20261104`
- Development: seeds 67201/67202; corrected amendment and rerun recorded
- Fresh/audit: sealed after exact native-control alias

## H — Hypothesis

Precomposing input-gated LoRA expert coefficients over a rank-4 shared factor bank preserves output with fewer bytes and active MACs than evaluating all experts; Mirror must beat native coefficient precomposition.

## T — Treatment and controls

Synthetic D=32, E=16, R=4, 512 examples; residual RMS 0/.05; two seeds. Compared full bank, full top-2 expert execution, shared rank-4 factor precomposition, native coefficient precomposition, and paired rank-2 private residual controls at rho=.05. The initial dense-atom/unevaluated-top2 attempt is preserved in `source/attempts/initial_dense_atom_unpruned_top2/`; corrected execution and private controls were amended before rerun. Tests replay metrics from serialized payloads.

## D — Decision

FAIL for Mirror-specific value. At rho=0, code precomposition is exact at 8.0KB versus 67.4KB full and 950,272 versus 9,175,040 MAC proxy. At rho=.05, shared-only error is .010–.012; rank-2 residual changes it to .009–.011 while doubling payload to ~16.2KB and MAC proxy to ~2.0M. Native coefficient controls exactly match outputs and MACs; their filenames/keys make files 14B larger, which is not a functional Mirror advantage. Top-2 full expert routing has .24–.31 NRMSE. Fresh stayed sealed.

## C — Strongest counter-hypothesis

Linearity lets ordinary low-rank coefficient aggregation perform the same function and compute. The large frontier gain against a dense bank is shared low-rank factorization, not Mirror-specific freedom.

## U — Unconfirmed

Natural trained adapter quality, learned gates, optimized GPU kernels, and hardware speedup remain untested. MAC proxy is not latency.

## Evidence labels

- Fact: payload-replayed output errors, actual NPZ bytes, and MAC proxies appear in `RESULTS_CORE.csv` and `source/development.json`.
- Interpretation: precomposition is efficient for aligned LoRAs; at tested residual level private rank-2 state was not needed to pass .02 output NRMSE.
- Hypothesis: Mirror-specific gains require a composition mechanism beyond linear code aggregation.
