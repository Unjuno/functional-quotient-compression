# Multi-stream Mirror role family diagnostic (2026-11-04)

## Decision

Operationally pause MA-681 and MA-682 in their current forms. Keep them UNTESTED. Continue to distinct mechanisms outside role coefficient factorization.

## Evidence

- MA-679: rank-1 stream-role embedding codes reconstructed held-out token embeddings exactly at about 54% of independent table bytes with zero measured cross-stream leakage. Native rank-1 role-conditioned shared-basis embeddings exactly matched the function and compute.
- MA-680: rank-1 stream-role QKV views reconstructed causal attention exactly at about 56% of independent bytes with zero measured future/cross-stream leakage. Native shared-basis QKV coefficients exactly matched the function and compute.
- MA-685 overlaps MA-640's RRDA dual-role write/suppress code mechanism; defer unless a distinct mechanism is specified.

## Interpretation

Both tested applications reduce to ordinary shared-basis role coefficients. The observed storage gains are real for the synthetic aligned functions, but neither supports Mirror-specific attribution. The zero leakage results validate the isolated toy implementations, not a trained multi-stream Transformer.

## Resume condition

MA-681 may resume with an explicit parallel execution/state reuse schedule and measured throughput gain over native stream/time conditioning. MA-682 may resume with a concrete learned stream-specific utility, leakage or private allocation mechanism that beats native low-rank plus residual controls. Shared basis plus role factorization alone is insufficient.

The next independent P0 is MA-686, product-key retrieval of Mirror edit codes, which tests retrieval cost, collision behavior and wrong-scope edits.
