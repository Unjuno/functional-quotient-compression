# MA-401 — FiLM versus Mirror feature conditioning

Status: **FAIL at the frozen development screen**.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Base commit: `c935a90` (`research/mirror-application-worker-ready-20261007`).
Prior art: PA63, FiLM conditioning.

## H — Hypothesis

For feature transforms aligned with a fixed family of pairwise Givens rotations, eight per-context Mirror angles recover held-out targets more accurately than FiLM and rank-one residual controls with fewer serialized bytes. As targets move outside that family, private residual state is necessary. The preregistered full gate additionally requires competitive runtime and a byte ratio at or below 0.10 versus the independent affine upper control.

## T — Treatment and controls

A shared frozen 16-dimensional nonlinear feature block was used across eight contexts and five target mixtures (`alpha=0` dense-private through `alpha=1` Givens-aligned). Controls were shared identity, FiLM, rank-one affine residual, Mirror, Mirror plus rank-one residual, and independent full affine transforms. Each method ran 400 Adam updates on development seeds 40101 and 40102, using 4096 examples per context and 1024-example optimizer batches. Fresh seeds 40111–40113 were not accessed.

Serialized inference payloads are deterministic NPZ files with FP16 weights/codes and UTF-8 metadata. Every shared tensor, code, coefficient, and metadata byte is included. `source/verify.py` reconstructs outputs from those payloads and replays all 60 metric rows. Runtime is eager single-thread CPU, 10 warmup and 25 timed iterations at batch 1024.

## D — Decision: FAIL

At the aligned endpoint (`alpha=1`):

- Seed 40101: Mirror NRMSE 0.000411 versus FiLM 0.589127 and rank-one 0.616020. Payloads were 2,590 / 4,557 / 22,714 bytes (Mirror / FiLM / independent). Mirror throughput was 0.149x FiLM; Mirror/independent payload was 0.114x.
- Seed 40102: Mirror NRMSE 0.000394 versus FiLM 0.525009 and rank-one 0.534795. Payloads were 2,606 / 4,580 / 22,661 bytes. Mirror throughput was 0.113x FiLM; Mirror/independent payload was 0.115x.

Thus quality and Mirror/FiLM byte gates passed, but both runtime and independent-byte gates failed in each seed. Per preregistered gate, fresh testing did not proceed.

Across alpha, Mirror NRMSE degrades as the target leaves the Givens family (about 1.21 at alpha=0 to below 0.001 at alpha=1). Mirror plus rank-one residual improves the off-family region but remains far from the independent affine upper control; the independent control is near 0.02 or better. This supports the narrow statement that private degrees of freedom are needed for these unrelated targets.

## C — Strongest counter-hypothesis

The target family was deliberately aligned with the chosen Givens coordinates, so the quality advantage demonstrates representational alignment rather than general functional capacity. The eager CPU implementation of the pairwise trigonometric transform also has substantial avoidable overhead; an optimized kernel could change runtime, but no fresh claim is made from this screen.

## U — Unknown

Natural task value, optimized-kernel throughput, and fresh-seed replication remain untested. This result is a narrow synthetic development-screen FAIL, not evidence that Mirror feature conditioning is universally ineffective.

## Fact / Interpretation / Hypothesis

**FACT:** Both development seeds met quality and Mirror/FiLM payload gates and missed runtime and Mirror/independent payload gates. Serialized-payload replay passed for all six methods and all five alpha points. Fresh remained sealed.

**INTERPRETATION:** Eight Givens coordinates compactly encode this aligned family, but the chosen implementation is too slow and its total payload does not reach the strict independent-byte threshold. Off-family performance requires private state.

**HYPOTHESIS:** Better kernels or a wider shared object could improve the storage/runtime frontier, but that requires a separately preregistered experiment.
