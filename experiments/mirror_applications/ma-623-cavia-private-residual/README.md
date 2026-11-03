# MA-623 — CAVIA shared context plus private residual

Status: SCREENING; protocol to be frozen before development. Prior art: PA129. Dedicated branch: `research/ma-623-cavia-private-residual-20261103`.

## H

In a compositional context family with pair-specific interactions, a shared factorized CAVIA context plus a low-dimensional private residual will recover held-out task functions at lower actual payload bytes than independent full contexts. The residual dimension needed to meet quality will reveal where sharing stops helping. Compare a native low-rank PCA residual with the same-dimensional direct task residual.

## T

Continue the MA-622 synthetic 16-input / 8-context linear mechanism with 4×4 factor pairs, diagonal held out, 16 support and 128 audit examples per task. Use aligned and private-interaction worlds and development seeds 62301/62302. Fit shared factors only on 12 training combinations. For held-out tasks adapt private residual coordinates from support examples. Sweep residual ranks 0, 1, 2, 4, 8. Controls: independent 8D CAVIA context fitted from each held-out support; direct residual coordinates; zero residual. Store complete `.npz` payloads including shared base, basis, shared factor codes, residual basis, per-task residual codes, factor indices, and metadata; report actual bytes and support-solve cost. Fresh 62311–62313 remain sealed unless development quality/storage pass and PCA residual improves over its direct-coordinate native control at comparable bytes/compute.

## D

Pending development.

## C / U

Counter-hypothesis: ordinary low-rank CAVIA context adaptation already explains any private-residual benefit. Natural task families, meta-training, noisy support, and language quality are outside this mechanism screen.

## Fact / Interpretation / Hypothesis

Keep measured outcomes separate from interpretation and extrapolation in `STATUS.md`.
