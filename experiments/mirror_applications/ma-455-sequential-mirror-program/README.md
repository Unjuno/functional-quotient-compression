# MA-455 — Sequential Mirror program over one shared block

Status: **FAIL under registered quality/byte gates**
Evidence lane: SEQUENTIAL COMPOSITION / ORDER SENSITIVITY / QUALITY / BYTES / COMPUTE
A1 freeze: `eb08b0fd`; valid fresh worlds 45520–45522. Initial worlds 45510–45512 are excluded because the rank-one residual control was gradient-dead under zero initialization.

## H — Hypothesis

One shared 2×2 block reused at two ordered depths, with one Givens Mirror angle per step, can approximate noncommuting two-block functions with at least 25% lower N=20 payload and at most 10% relative NRMSE penalty versus independent blocks.

## T — Test

Teacher tasks were `y=A₂A₁x`, where each 2×2 block was a rotation followed by a random shear. Each task used 64 support inputs and 512 query inputs; three fresh worlds × three seeds × 64 tasks. Models used 500 Adam updates at lr 0.03: tied `W·W`, Mirror `(R(θ₂)W)(R(θ₁)W)`, shared W plus per-step rank-one residuals, and two independent matrices. Actual serialized payloads, support fitting wall, query wall, MAC proxy and order-swapped error were recorded at N=1/20/64.

A1 excluded the first fresh attempt after finding both rank-one factors initialized to zero. Nonzero factor initialization and replacement worlds were fixed before valid fresh access.

## D — FAIL

**Fact:** At N=20, across valid fresh runs, mean NRMSE was 0.000617 tied, 0.0000154 Mirror, 0.00000551 low-rank residual, and 0.0000000311 independent. Payload bytes/task were 133.25 tied, 177.65 Mirror, 273.25 low-rank residual, and 193.85 independent. Mirror saved 8.4% vs independent, missing the 25% gate, and its error was over 400× the independent error. Mean commutator norm was 0.516 and order-swapped NRMSE 0.319, confirming order sensitivity. Mirror averaged 0.264s support-fit wall per 64-task batch; independent averaged 0.169s.

**Interpretation:** Mirror improves on tied depth in two of three worlds, but does not beat independent quality and does not save enough storage. At N=20 the tied block is smaller and its mean quality is already strong; one world had lower tied error than Mirror. The correctly initialized rank-one residual improved quality over Mirror but cost more bytes.

## C — Strongest counter-hypothesis

The teacher products often lie near the tied-block representable set, so an ordinary repeated block already captures most of the useful composition; the remaining Mirror angles add state without recovering independent-block precision.

## U — Unknown

Higher-dimensional nonlinear blocks, longer programs, learned routing, natural tasks, and optimization to near-convergence remain untested. This is a synthetic 2D linear composition screen, not Transformer evidence.
