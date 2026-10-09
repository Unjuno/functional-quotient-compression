# Conditional modulation family pause — 2026-10-09 update

## New evidence: MA-551

MA-551 tested a direct hierarchical extension, shared FiLM plus per-function Givens angles, and used compiled C kernels that fuse FiLM and disjoint rotation application. The synthetic Givens-aligned functions were represented at 2,994 B with near-zero error, compared with 5,420 B for independent FiLM and 36,150 B for full affine. However, throughput was 0.570–0.693× independent FiLM in both registered development seeds, below the frozen 0.80× gate. A native Givens adapter had exactly the same serialized arrays and outputs. Fresh remained sealed.

## Family facts

| MA | Mechanism | Development failure |
|---|---|---|
| 401 | Feature-space Givens after shared features | 0.113–0.149× FiLM throughput; strict independent-byte gate missed |
| 403 | Token-position-generated Givens | 0.049–0.255× token-FiLM throughput; byte gates missed |
| 405 | Givens before shared FFN projection | One seed at 0.633× StyleGAN2 throughput, one at parity; runtime gate missed |
| 551 | Fused C shared-FiLM + Givens hierarchy | 0.570–0.693× independent-FiLM throughput; exact native Givens alias |

All use intentionally rotation-aligned synthetic targets and do not establish natural task value. MA-551 addresses the prior eager-kernel concern with a fused C implementation; the runtime deficit remains in this 16D/32-function screen.

## Pause and reopening criteria

Pause MA-552/553/554/556 and conditional-modulation Givens siblings pending a lower-operation functional representation or a measured optimized device kernel that clears the runtime gate. Further variants that still apply per-example Givens rotations repeat the same structural compute cost. Reopen only with a different operator family, fused device implementation, and a preregistered throughput comparison against native FiLM.
