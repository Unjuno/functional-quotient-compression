# MA-401 status

Status: **FAIL** at the frozen development screen; no fresh run.

## H — Hypothesis
Eight context-specific Givens Mirror angles recover aligned transformations at lower payload bytes than FiLM and rank-one controls. Off-family targets may need private residual state.

## T — Treatment
Shared frozen 16D nonlinear feature block; eight contexts; five transform mixtures; shared identity, FiLM, rank-one, Mirror, Mirror+rank-one, and independent affine controls. Development seeds 40101/40102, 4096 examples/context, 400 updates. Fresh seeds 40111–40113 remained sealed.

## D — Disposition
FAIL. Both development seeds met the alpha=1 quality and Mirror/FiLM byte gates. Both failed the throughput gate (Mirror/FiLM 0.149x and 0.113x) and the independent-control byte gate (payload ratios 0.114 and 0.115 versus a 0.10 maximum). No fresh testing.

## C — Strongest counter-hypothesis
The synthetic target family is aligned to the Givens coordinates, and eager trigonometric kernels are a costly implementation; the quality result may not generalize and runtime may improve with a fused kernel.

## U — Unknowns
Natural tasks, optimized kernels, and fresh replication remain untested.

## Evidence separation
FACT: metric and payload replay passed across six methods, two seeds, five alpha values. INTERPRETATION: aligned rotations admit compact codes but runtime and strict total-byte goals failed. HYPOTHESIS: a different implementation scale or kernel may improve the Pareto point; it requires a new protocol.
