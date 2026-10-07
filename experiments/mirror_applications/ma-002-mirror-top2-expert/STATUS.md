# MA-002 status

- Status: PROMISING
- Branch: `research/ma-002-mirror-top2-expert-20261007`
- Protocol/source freeze: `76caaf7`
- Development complete: yes; LR 0.01 selected only on world 20000
- Fresh worlds 20001–20003: complete for aligned and independent teachers
- Fresh replay: all 36 deterministic rows matched exactly
- Verification: 3 tests passed; serialized round-trip checked across 72 runs
- Registry: PROMISING

## H / T / D / C / U

- **H:** Four learned top-2 expert roles related by Givens views can be represented by one nonlinear FFN plus small role coordinates, retaining routed mixture quality at fewer bytes than untied experts; unrelated roles need private state.
- **T:** Six methods, two teacher modes, 1,200 updates × 64 examples; development world 20000 selected LR 0.01; frozen fresh worlds 20001–20003. See README/PROTOCOL.
- **D:** PROMISING: aligned useful-sharing gate passed 3/3, Mirror/untied MSE 0.223–0.403 and payload 0.331x. Mirror-specific Pareto gate failed vs tying (252B larger, ~2.03x active proxy); CPU throughput was 0.102x tying.
- **C:** The aligned teacher was generated from the Mirror Givens family; generic basis/hypernetwork controls remain untested.
- **U:** Natural language, convergence/capacity, optimized kernels and broad private-parameter boundary remain untested.

## Provenance

The frozen protocol's base-commit field typo is disclosed and corrected in `PROTOCOL_AMENDMENT.json`; no experimental condition changed.
