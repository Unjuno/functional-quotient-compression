# MA-709 — Toeplitz-like logical experts and heads

Status: **FAIL for Mirror-specific value; aligned compression feasibility observed**  
Evidence lane: MECHANISM/STORAGE/COMPUTE  
Base commit: `eae6e926`

## Hypothesis

H: One shared circular-convolution kernel plus per-task shift/gain coordinates can express aligned task maps at low serialized bytes; off-orbit kernel variation will identify where task-private parameters become necessary.

## Mirror insertion

> **Mirror insertion:** per-task cyclic shift and gain coordinates at a shared circular-convolution operator create logical task-specific maps without storing one dense matrix per task.

Controls: independently fitted dense matrices, independent Toeplitz kernels, hard tying, native shared circular-convolution shift/gain coordinates, and a DCT-diagonal structured operator. PA173 motivates Toeplitz/LDR matrices. PA45 motivates diagonal-plus-global-mixing structured methods; the DCT-diagonal comparator is not a full ACDC implementation.

## Amendment

The first attempt serialized one NPZ per task, multiplying container overhead and misrepresenting deployment payload bytes. Its CSV and payloads are retained under `source/attempts/per_task_container_accounting/`. The corrected run serializes one full model payload per method/seed/variation. Same seeds, task generation, controls, metrics, and gates were retained. Since initial audit outputs had been seen, corrected data are transparent same-seed accounting replays, not fresh confirmation. See [AMENDMENT.md](AMENDMENT.md).

## Results

**FACT:** D=64, 16 maps, 128 support and 128 held-out vectors per map; development seed 70901 and corrected audit replays 70911–70913; private kernel fractions ρ=0/.05/.2. At ρ=0, shared shift/gain payload was 1,348B and mean held-out NRMSE 0. Independent Toeplitz was 4,614B with mean NRMSE 0; dense was 262,662B with mean NRMSE 0. At ρ=.05, candidate NRMSE was .0488–.0533 across audits while independent Toeplitz remained <2.6e-8. At ρ=.2, candidate NRMSE was .1908–.2139; independent Toeplitz remained <2.6e-8. Hard tie NRMSE was about 1.42–1.48. DCT-diagonal mean NRMSE was about .80–.82 and payload 21,244B. Mirror and native shared shift/gain payload hashes and outputs were exact aliases. Candidate and Toeplitz arithmetic proxies were both 1,152 operations/vector; dense proxy was 4,096. No training updates were used.

**INTERPRETATION:** In this oracle-aligned synthetic world, one shared kernel plus small view codes reduced bytes by 70.8% versus independent Toeplitz and 99.5% versus independent dense. A 5% private kernel perturbation already broke the frozen exact-quality threshold, showing that private Toeplitz parameters are needed under modest task heterogeneity. Since the native shared circular-convolution control is byte/function identical, no Mirror-specific gain is established. The shared-kernel result is generic weight tying/conditioning feasibility, not capacity.

**HYPOTHESIS:** Shift/gain coordinates are useful only while task kernels remain close to one shared orbit; a private-kernel budget or stronger shared basis may move the frontier, but would need comparison with native low-rank/kernel-bank controls.

**BOUNDARY:** Synthetic circular-convolution regression with the teacher base kernel known by construction. No learned network, natural tasks, routing, capacity, or GPU-runtime conclusion. The first per-task serializer accounting was invalid and is preserved separately.

## H/T/D/C/U

- **H:** Shared Toeplitz kernel plus small task coordinates recovers multiple aligned maps at lower bytes; private task state is required off-orbit.
- **T:** Three corrected audit seeds, 16 maps, D=64, 128 support/held-out vectors, ρ=0/.05/.2; dense, independent Toeplitz, hard tie, native shift/gain, DCT-diagonal controls.
- **D:** **FAIL** for Mirror specificity. Aligned candidate reaches exact fit at 1,348B, but native control exactly aliases it. Quality breaks at ρ=.05; independent Toeplitz recovers the target.
- **C:** This is standard shared circular convolution with per-task shift/gain conditioning; moreover the teacher base was supplied, making the test favorable to the candidate.
- **U:** Whether a learned shared kernel bank or trained neural experts can preserve useful task quality under real task heterogeneity; whether fused kernels make the arithmetic proxy a runtime advantage.
