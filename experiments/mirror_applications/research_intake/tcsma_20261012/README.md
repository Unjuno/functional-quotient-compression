# MA-1194 — Task-conditioned sparse Mirror alignment (TCSMA-001/002)

**Scientific verdict: FAIL for incremental partial-matching value and Mirror-specificity.** Narrow development/negative controls show sharing+private residual can reduce payload, but not preserve independent model quality. This is registered on an **isolated research/staging branch**, NOT canonical worker-ready, and main is untouched.

## Hypothesis and native comparison
Try to learn a low-description task-specific group operator Q, sparse top-2 coefficient A over a shared D, and rank-one private residual E, with an additional capped nearest-neighbor partial alignment penalty between ridge-estimated per-task canonical rows. The original general TCSMA proposal includes exact partial correspondences and possibly nonlinear functions; **this first CPU implementation is only a capped non-bijective nearest-neighbor surrogate and a synthetic linear teacher.**

All parameters are fitted jointly from train (X,Y) examples; the hidden teacher support and true transform angles are not fed to the learner. The ridge-estimated W for matching uses training labels only. Each task has six output vectors in dimension 24; 12 tasks, r=8 shared atoms, k=2 active per row, 3 plane-rotation codes, optional private rank1. Train: 128 examples per task, 450 Adam updates; test: 512 new examples per task, FP32 CPU one thread.

**Direct controls:** independent per-task ridge (strong quality control), native unconstrained shared low-rank dictionary, ordinary fixed-coordinate sparse dictionary, rotation+private without matching, angle View+ridge auxiliary. A native group-rotated sparse dictionary with identical A,D,Q,E is **an exact functional and byte-level alias**, precluding Mirror-specific novelty.

## First three development worlds, 3 seeds (1001..1003)

| Synthetic family | Native shared dense NRMSE | View + private without matching | With TCSMA partial matching |
|---|---:|---:|---:|
| Exactly rotation-aligned | 0.009753 | 0.005936 | 0.004807 |
| Partial: 1/6 output row off orbit | 0.088701 | 0.034009 | **0.021833** |
| Unrelated independent 6/6 | 0.502270 | 0.383124 | 0.383124 |

Development G1 FAIL (the aligned per-seed margin >=0.01 was not met). G2 numeric quality/byte gate passed on partial. Original fresh 1101..1103 remain **sealed**; no retrospective gate repair.

## Separate prospectively frozen 5-seed partial replication

The dev partial result motivated a **distinct versioned confirmation protocol** preregistered on GitHub at `2ec823ba788a4456d0a6de9f677945ab0055deed` **before opening** new fresh [2101..2105]. The science source and original train protocol hashes remained unchanged. New 2-seed unrelated negative controls were also prespecified.

| Fresh seed | View+private no match (test nMSE) | Match+private TCSMA (nMSE) |
|---:|---:|---:|
| 2101 | 0.026071 | 0.026342 |
| 2102 | 0.027008 | 0.042755 |
| 2103 | 0.030165 | 0.030182 |
| 2104 | 0.026030 | 0.028265 |
| 2105 | 0.022631 | 0.034275 |
| **Mean** | **0.026381** | **0.032364** |

**Primary H1 FAIL:** TCSMA improved test nMSE in 0/5 newly evaluated seeds; the predeclared pass required >=4/5 positive gains and >=0.005 mean improvement. Mean (no-match minus match) = -0.005983; seed-bootstrap 90% descriptive interval [-0.01101,-0.000955]. Same data-generating family as development, independent synthetic seeds—not natural task datasets.

**Secondary H2 limited numeric PASS:** entire 12-task inference archive mean 4,662.2B vs ridge 6,859.6B, about 32.0% smaller; absolute TCSMA test nMSE <0.06 in 5/5. **But independent ridge nMSE ~0.000039, so this is NOT quality-preserving compression or Pareto superiority.** Matched group's full fit wall time averaged 7.10s vs 3.07s without matching. Folded batch64 runtime passes the relative +25% tolerance, not a novel speedup.

**Negative worlds:** TCSMA matches no-match results exactly at nMSE .3784/.3682, and selects zero inter-task matches; independent ridge near 0.000057. The training objective does not discover unrelated capabilities.

**Mirror-specific G4 FAIL:** same source-trained group-equivariant sparse dictionary and native coefficient/angle representation produce exactly the same logical function; changing terminology does not add independent degrees of freedom.

## Integrity and provenance
- 11/11 standalone unit tests PASS; 72 development + 28 confirmation inference NPZs reloaded and all numeric metrics recomputed with max observed difference 0; 24 earlier calibration payloads separately retained.
- Two new-fresh models fully retrained from initial seed and optimizer state; both final serialized SHA256 match exactly.
- Original TCSMA001 fresh [1101,1102,1103] unopened. Post-hoc angular diagnostics do not update preregistered scientific outcomes.
- Hardware: Intel Xeon Platinum 8370C ~2.80 GHz clock snapshot 2793 MHz, 4-core cgroup quota, memory 4 GiB, PyTorch 2.10.0+cpu, Python3.13.5, one Torch thread, FP32; batch64 training / batch1 and64 inference, no GPU.
- Full code, all 124 inference payloads, prior logs, summary, hashes and exact replay are in the conversation attachment `TCSMA_2026-10-12_FULL_EVIDENCE.zip` (SHA256 `5ade624c3ac130e921f9567b0ee33e860b6ef49154df47a0239b1e8a148b42f4`). Smaller `TCSMA_2026-10-12_SOURCE_REPORTS.zip` SHA256 `5af2a1a8e1d838d54b57b0e9aed42f27979f2b0244d8137538324523c5b5a595`. The **runnable source and raw payload archives are conversation artifacts, not yet uploaded into GitHub**; this branch stores protocol, verified summary and provenance.
- Full Japanese report and complete H/T/D/C/U: `REPORT_JA.md` in archive.
- Closest prior: Soh, *Group Invariant Dictionary Learning*, IEEE TSP 2021, DOI 10.1109/TSP.2021.3087900; Hu et al., *Task-aware Orthogonal Sparse Network*, ICML 2024, https://proceedings.mlr.press/v235/hu24b.html.

## Next falsifiable change
A new protocol should replace capped bidirectional 1-to-many nearest-neighbor proxy with one-to-one partial matching plus unmatched dummy and confidence-calibrated task-loss weighting, first show gains beyond **native group-invariant dictionary learning** at matched training cost and then test real independent nonlinear task families. Do NOT silently retune 2101..2105.
