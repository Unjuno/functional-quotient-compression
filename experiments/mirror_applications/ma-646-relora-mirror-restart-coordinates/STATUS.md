# MA-646 status

- Status: FAIL
- Branch: `research/ma-646-relora-mirror-restart-coordinates-20261104`
- Base commit: `7cc969a3`
- Protocol frozen before development: yes
- Development complete: yes (64601, 64602)
- Fresh/audit opened: no; frozen failure gate reached

## H / T / D / C / U

**H:** Reusing one rank-4 basis across restart phases caps cumulative update rank at four; per-phase addresses into a shared overcomplete basis can restore rank growth, but must beat merged ReLoRA and native block-ID controls in bytes to support a Mirror-specific claim.

**T:** Closed-form synthetic 32x32 rank-16 target, decomposed into four orthogonal rank-4 phase updates. Two development seeds. Compared independent ReLoRA phase factors merged into final weights, a static rank-4 shared basis, a shared rank-16 dictionary plus phase codes, and native dictionary block IDs. Actual inference payload and restart-state NPZ bytes, relative Frobenius error, numerical rank, reconstruction MAC proxy and serialization wall time measured. Zero optimizer updates; this is a representation test, not a training benchmark.

**D:** FAIL. ReLoRA and shared dictionary recover rank 16 with relative error 7.59e-8 on both seeds. Static basis has rank 4 and relative error 0.837. Shared dictionary restart state is 8,780B, exactly equal to ReLoRA phase factors; its complete inference payload is 8,780B vs 8,297B for merged ReLoRA. Native block IDs use the exact same basis, codes, coefficients, outputs and bytes as Mirror. Fresh seeds 64611–64613 remain sealed.

**C:** A fixed low-rank span cannot accumulate novel directions, while address selection over a shared overcomplete dictionary is standard codebook/block selection. ReLoRA's merge produces a dense final matrix that is cheaper to serialize here than keeping the dictionary and coefficients.

**U:** No optimization convergence, optimizer-state behavior, neural network training, or practical ReLoRA runtime was tested. Fresh results are not established.

## Fact / Interpretation / Hypothesis

- **Fact:** 8 development rows (2 seeds x 4 methods); all serialized byte counts and reconstruction metrics replay; three tests pass; protocol hash matches freeze.
- **Interpretation:** Basis reuse alone prevents cumulative rank growth. Addressing distinct shared subspaces restores rank but does not reduce actual bytes or outperform native block IDs in this task.
- **Hypothesis:** Temporal low-rank composition may be useful when task updates share compressible subspaces, but it needs a benefit beyond ordinary indexed basis blocks and a quality-matched payload frontier.
