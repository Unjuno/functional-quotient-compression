# MA-006 status

- Status: PROMISING for the registered expert-choice sharing gate
- Branch: `research/ma-006-expert-choice-mirror-20261007`
- Protocol/source freeze: `d1d0d76`
- Development selected LR 0.003 only on world 60000
- Fresh worlds 60001–60003 complete for both teacher modes
- Fresh replay: all 48 deterministic rows match exactly
- Verification: 3 tests passed; serialization round-trip checked across 96 runs
- Registry: PROMISING

## H / T / D / C / U

- **H:** Shared nonlinear expert parameters plus Mirror role views can retain capacity-balanced expert-choice quality at lower payload bytes; unrelated roles may require private state.
- **T:** Eight methods, two teacher modes, 1,200 updates × 64 examples; each batch has 16 examples per role and each expert selects 16 tokens. Development world 60000 selected LR 0.003; fresh worlds 60001–60003.
- **D:** PROMISING for expert-choice sharing: quality/coverage/bytes passed 3/3; Mirror/full-EC MSE 0.954–1.006 and 0.333x payload. Mirror-specific control gate missed, token-choice had 100% coverage and 2.50–3.42x lower MSE. CPU throughput 0.274x tied.
- **C:** Expert-choice overlaps/no-route pattern explains much of quality loss; token-choice dominates on this task.
- **U:** Language-scale behavior, fallback, larger pools, fixed-byte capacity and optimized dispatch remain untested.
