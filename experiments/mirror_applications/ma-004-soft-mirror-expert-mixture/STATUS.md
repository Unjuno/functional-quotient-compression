# MA-004 status

- Status: PROMISING
- Branch: `research/ma-004-soft-mirror-expert-mixture-20261007`
- Protocol/source freeze: `74d00b2`
- Development selected LR 0.01 only on world 40000
- Fresh worlds 40001–40003 complete for both teacher modes
- Fresh replay: all 36 deterministic rows match exactly
- Verification: 3 tests passed; serialized model round-trip checked across 72 runs
- Registry: PROMISING

## H / T / D / C / U

- **H:** One nonlinear FFN plus four learned Givens views can recover the full softmax-weighted outputs of four aligned expert roles at fewer bytes than untied dense soft MoE; unrelated roles need private capacity.
- **T:** Six methods, two teacher modes, 1,200 updates × 64 examples; development world 40000 selected LR 0.01; fresh worlds 40001–40003.
- **D:** PROMISING. Aligned gate passed 3/3; Mirror/untied MSE 0.297–1.030 and payload 0.331x. Mirror-specific Pareto gate failed vs hard tying: 252B more and ~4x active MAC proxy. CPU throughput 0.086x tied.
- **C:** Teacher functions were deliberately Givens-aligned; a generic shared basis or generated control could narrow the gap.
- **U:** Language transfer, fixed-byte capacity, broader role counts and optimized kernels remain untested.
