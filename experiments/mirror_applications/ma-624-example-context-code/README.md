# MA-624 — Infer CAVIA context from support examples

Status: SCREENING; protocol frozen before development. Prior art: PA130 Conditional Neural Processes. Branch: `research/ma-624-example-context-code-20261103`.

## H

For the same 4×4 synthetic factor task family, infer a compact factor-addressed Mirror code from support examples and test held-out factor combinations. It is useful only if it improves the storage/compute/quality frontier over generic context aggregation or direct CAVIA support adaptation. The factor identity is not provided to the encoder.

## T

Use 16-input / 8-context linear functions, 16 support and 128 disjoint audit examples per task. Train the factor codebook on 12 combinations; diagonal pairs are held out. Development seeds 62401/62402; fresh 62411–62413 sealed. Compare: zero context; direct CAVIA least-squares context from support; generic pooled residual context (CNP-style); Mirror code inference by fitting the held-out support against the learned factorized context family; and independent per-task codebook upper control. Include full serialized model/codebook/index payload bytes, support compute and wall time. No audit examples in fitting.

## D

Pending.

## C / U

Counter-hypothesis: direct CAVIA support fitting is the optimal generic linear conditioner and the Mirror codebook adds parameters and solve cost. This synthetic screen does not establish natural few-shot or meta-learning performance.
