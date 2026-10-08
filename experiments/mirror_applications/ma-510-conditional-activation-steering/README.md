# MA-510 — Conditional activation steering with factorized condition and behavior Views

Status: SCREENING; development not run yet.
Branch: research/ma-510-conditional-activation-steering-20261008
Prior art: PA101, Conditional Activation Steering.
Base commit: c03c1da669b812d74f19b606c3264750b1edf42b.

## H — Hypothesis

On two frozen synthetic contextual-activation worlds with 16 condition classes and 8 reusable behaviors, a rank-4 factorized condition/behavior Mirror will preserve conditional outputs (event recall >=.95, conditional behavior accuracy >=.95), hold hard-negative false-trigger rate <=.01, and use <=.65x the actual serialized bytes of explicit CAST condition and behavior vectors.

Mirror insertion: factor condition and behavior intervention vectors into two shared bases, and let a condition address choose a behavior code only when the condition score passes a calibrated gate. All basis, code, address, threshold and serialization state is paid.

The strongest attribution control is native PCA/SVD of those same banks. It has the same function class and may alias exactly. Explicit CAST is the native condition-vector similarity gate and explicit intervention bank; independent condition-behavior vectors are an upper storage control.

## T — Frozen protocol

See PROTOCOL.json and hashes in freeze.json. Two dev seeds (51001/51002), 64 dimensions, 16 conditions, 8 behaviors, 128 support contexts per condition, separate gate-calibration data, and 256 positive plus 8,192 hard-negative evaluation contexts per condition world. The fixed rank grid is condition {2,4,8,16} × behavior {2,4,8}; the primary point is rank4/rank4. Thresholds use calibration negatives only. Fresh seeds 51011–51013 are locked and unopened.

## D — Decision

Pending frozen development run.

## C — Strongest counter-hypothesis

Native condition-vector similarity plus ordinary PCA of intervention banks already provides the same routing and factorization. Apparent conditional-view gains may therefore be standard shared-basis compression, while false triggers and low-rank reconstruction error erase the apparent benefit.

## U — Scope limits

No pretrained LM checkpoint is present in this checkout. This screen uses a synthetic frozen activation mechanism, not natural-language prompts or a semantic reproduction of CAST. Even a positive screen would require a separately registered natural-prompt/frozen-LM experiment before making language or safety claims.

## Evidence classification

- Facts: to be populated from deterministic payload replay and the frozen development seeds.
- Interpretation: conditional gate and storage/compute tradeoffs are mechanism-level evidence only.
- Hypothesis: learned condition/behavior codes may compose useful logical behaviors beyond explicit CAST vectors; native PCA or ordinary gates may explain the full effect.
