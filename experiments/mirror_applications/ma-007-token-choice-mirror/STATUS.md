# MA-007 status

- Status: PROMISING for the registered token-choice sharing gate
- Branch: `research/ma-007-token-choice-mirror-20261007`
- Protocol/source freeze: `76cf864`
- Development selected LR 0.01 only on world 70000
- Fresh worlds 70001–70003 complete for both teacher modes
- Fresh replay: all 48 deterministic rows match exactly
- Verification: 3 tests passed; serialization round-trip checked across 96 runs
- Registry: PROMISING

## H / T / D / C / U

- **H:** Token-choice Mirror can restore every token's expert assignment and retain compressed aligned function quality vs expert-choice.
- **T:** Eight routing/expert methods, two teacher modes, 1,200 updates × 64 examples, fresh worlds 70001–70003. Development selected LR 0.01.
- **D:** PROMISING: quality/storage passed 3/3, Mirror/full token MSE 0.766–0.769 and payload 0.334x; coverage 100% vs 0.818–0.827 expert-choice. Hard tying remains 252B smaller; CPU throughput is 0.233x tied.
- **C:** Givens-aligned synthetic teacher is favorable; generic bases and language data are untested.
- **U:** Near-converged capacity, natural language, optimized inference, larger role counts and full private-state boundary remain untested.
