# MA-899 — P-4DGS predictive residual as Mirror temporal code

Status: **UNTESTED — blocked before model evaluation**
Evidence lane: NEURAL_GRAPHICS / TEMPORAL-COMPRESSION / RATE-DISTORTION / RANDOM-ACCESS / RUNTIME
Base commit: `414f9a5b9e5f854d7832d4bcb5b99395492905d9`

## H — falsifiable hypothesis

Adding a compact block-specific temporal coordinate to one shared spatio-temporal anchor predictor can recover useful logical frame residual functions with fewer actual serialized bits than independent frame corrections, while matching or improving the native P-4DGS rate-distortion and random-access/frame-throughput frontier. It is falsified if ordinary predictive residual entropy coding matches the result, Mirror misses the quality gate, or its complete bitstream/compute costs erase the savings.

## Prior-art delta

PA251 already uses anchor-based spatio-temporal prediction, adaptive quantization, and context-based entropy coding. These are mandatory direct controls; a Mirror-specific gain requires improvement beyond them and a byte-near native low-rank/residual-coordinate control.

## Execution and limits

See `EXECUTION_BLOCKER.md` and `source/ENVIRONMENT.md`. The bounded public-source search did not identify an official code release; there is no implementation or benchmark data in the workspace and the runtime has no CUDA device. No predictive codec or model was run. This is a feasibility review, not a negative quality result.

## Report

H/T/D/C/U are recorded in the blocker report. Fact: no model execution. Interpretation: registered frontier is not established. Hypothesis: Mirror value over native prediction/entropy coding remains untested.
