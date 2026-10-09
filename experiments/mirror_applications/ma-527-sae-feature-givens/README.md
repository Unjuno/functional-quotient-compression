# MA-527 — Givens Views over SAE feature activations

Status: **FAIL**. Dedicated branch: `research/ma-527-sae-feature-givens-20261009`. Frozen protocol hash: `63ea2cee5e40b5428070ef683a51de64a99c50617f67e81161c2554563b13af2`. Prior art: PA96 and PA102.

## H — Hypothesis

One shared 16-value SAE code plus eight task-specific Givens angles will preserve held-out causal quality within 0.10 gold-candidate log-probability nats and 0.05 accuracy of explicit function vectors (FVs), use at most half their payload bytes, and beat both an equal-size pairwise-gain control and global OMP-16 by at least 0.10 nats on both development seeds.

## T — Protocol and execution

Pinned Pythia-70m and the same pretrained 4x layer-3 SAE as MA-526. Tasks 0–11 selected a fixed pool of 16 SAE atoms and fit one shared vector with 2,000 full-batch Adam updates. Tasks 12–15 were held out; only their eight Givens angles or equal-size pairwise gains were fit (500 updates). Controls were no intervention, explicit FP32 FV, shared SAE vector without transform, native pairwise gains, and global signed OMP-16. Development seeds were 52701 and 52702. Fresh seeds 52711–52713 remain locked. Protocol and task splits were unchanged after freeze.

The eight Givens pairs were fixed as `(0,1), (2,3), …, (14,15)`. We charged actual uncompressed NPZ bytes for all indices, codes, angles/gains and metadata. We report incremental state over a preloaded SAE and standalone Pythia+SAE deployment. Wall times include per-method pool/fit or OMP coding and candidate inference; support-FV extraction is common. Operation proxies separate SAE pool work, code fitting or OMP, and candidate-sequence work.

## D — Decision

**FAIL; fresh data stays sealed.** The byte and native-control margins pass, but explicit-FV causal quality fails in both seeds.

| Seed | Method | Payload B | Accuracy | Gold log-prob | Δ vs explicit FV | Δ vs native gains | Δ vs global OMP-16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 52701 | Explicit FV | 34,826 | 0.2500 | −10.568500 | — | — | — |
| 52701 | Givens View | 3,432 | 0.1563 | −12.901594 | −2.333094 | +0.204396 | −1.598107 |
| 52702 | Explicit FV | 34,826 | 0.1875 | −10.067564 | — | — | — |
| 52702 | Givens View | 3,432 | 0.2188 | −12.674618 | −2.607053 | +0.224922 | −1.559831 |

Givens uses 9.85% of the explicit FV payload as incremental state. Its total standalone deployment is **4,172,997 B larger** than Pythia plus explicit FVs because the 4,204,391 B SAE is required. The Givens payload is 4 B larger than the native gain payload (3,428 B). Candidate wall time including support extraction was 24.406 s vs 21.495 s for gains and 18.271 s for explicit FV on seed 52701; on seed 52702 it was 19.382 s vs 16.642 s and 13.359 s, respectively. Times are CPU measurements in this container, not optimized-kernel throughput claims.

## Evidence classes

**Facts:** five tests pass. Deterministic replay matched ten paid NPZ payload hashes, core metrics, task splits and selected pool exactly; max core-metric difference was zero. Fresh seeds were not accessed. Three pre-metric implementation failures are preserved in `IMPLEMENTATION_AMENDMENT_1.json`; the frozen protocol hash did not change.

**Interpretation:** orthogonal feature rotation modestly improves gold likelihood over pairwise gains on both seeds at nearly identical bytes, but that margin does not recover the explicit FV behavior. Incremental compression is large only when the SAE is treated as already paid; standalone storage is worse.

**Hypothesis:** the selected 16-feature SAE subspace may not express these support-derived task FVs well, even after learned orthogonal views. A behavior-coefficient code over a broader shared feature basis is a distinct follow-up, not evidence of general SAE steering capacity.

## C — Strongest counter-hypothesis

The apparent Givens benefit may be an optimizer/coordinate parameterization advantage over direct scalar gains rather than a Mirror-specific effect. Both methods remain far below explicit FV, and global sparse coding is simpler and more accurate.

## U — Limits

Only one SAE checkpoint, one small causal LM and four held-out relation tasks were measured. No fresh-seed generalization, near-convergence capacity, natural generation, hardware-optimized runtime, or broad feature interpretability claim is established.
