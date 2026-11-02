# MA-551 — FiLM-to-Mirror hierarchy

Status: **FAIL** (runtime gate and native Mirror-specific attribution)
Branch: `research/ma-551-film-mirror-hierarchy-20261009`
Prior art: PA106 (FiLM), PA63 (FiLM), PA65 (StyleGAN2 modulation)

## H — Hypothesis

Shared FiLM state plus small per-function Givens Views can recover context-specific affine functions at lower actual bytes than independent FiLM or full affine state, while a fused optimized kernel retains at least 0.80× independent-FiLM throughput.

## T — Execution

Synthetic width-16 functions across 32 contexts. The teacher generates one shared diagonal affine transform followed by eight pairwise-disjoint per-context Givens rotations. Controls are tied FiLM, oracle least-squares independent FiLM, shared FiLM plus Mirror Givens, the same native shared-FiLM plus Givens adapter, shared FiLM plus rank-4 private residual, and independent full affine. No optimizer or training is used; this is an oracle storage/quality/runtime mechanism screen. Development seeds were 55101/55102. Fresh seeds 55111–55113 were sealed by the preregistered runtime gate miss.

All methods use compiled C kernels. Mirror fuses shared FiLM and eight disjoint rotations in one pass; trigonometric state is derived from the stored angles during model load. Runtime uses 10 warmups and 30 timed banks of 32×1024 examples. Actual NPZ bytes include every method array and metadata.

## D — Decision: FAIL

Both development worlds meet quality and payload gates: Mirror NRMSE is about `1.2e-16`, payload is 2,994 B versus 5,420 B for independent FiLM and 36,150 B for full affine. But fused Mirror throughput is only 0.693× and 0.570× independent FiLM, below the frozen 0.80 gate. The ordinary native Givens adapter has byte-identical parameter arrays and output-identical metrics, so there is no Mirror-specific attribution. Per protocol, no fresh worlds were opened.

The rank-4 shared/private control does not reconstruct the structured rotations accurately (max per-function NRMSE 0.311–0.350) and is slower. Full affine is an exact upper control but costs 12.1× Mirror bytes. This is an aligned synthetic demonstration only.

## C — Strongest counter-hypothesis

The target functions were constructed from the same disjoint Givens family as the treatment. The task-conditioned transform has a conventional native parameterization that exactly aliases the Mirror code, and even after fusing the kernel, the extra rotations cost 31–43% of FiLM throughput.

## U — Unknown

Natural-task quality, learned function codes, larger widths/models, fused GPU kernels, and energy use are untested. The explored C kernel demonstrates that a simple CPU fusion does not clear the runtime gate; conditional-modulation candidates remain paused pending a more efficient operator or materially different function family.

## Fact / Interpretation / Hypothesis

**FACT:** Quality and byte gates pass; throughput gates fail in both development worlds; native Givens code exactly aliases the Mirror arrays and outputs; replayed six NPZs are byte-identical; fresh remained sealed.

**INTERPRETATION:** Shared FiLM plus per-context rotations is a compact way to represent this chosen family, but the benefit is ordinary structured parameterization and the tested kernel trades speed for bytes.

**HYPOTHESIS:** A different low-compute View geometry could retain structured role variation without the measured rotation cost. This needs a new non-Givens hypothesis.
