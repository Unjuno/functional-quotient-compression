# Function-vector code family diagnostic — 2026-10-09

## Decision

Pause unchanged static shared-PCA/linear-code variants for function-vector compression and arithmetic. Continue with MA-521 because demonstration-to-code compilation changes how the address is inferred and has a distinct direct-ICL control.

## Facts

- **MA-516:** Explicit support-derived FVs improved held-out gold candidate log probability by 2.62/2.97 nats over no intervention on the registered Pythia screen. Rank-four shared PCA lost 0.535/0.606 nats versus explicit FVs and exactly matched the native PCA control. Its actual payload was 12,688 B versus 34,454 B explicit. Fresh seeds stayed sealed.
- **MA-517:** No-intervention identity-composition accuracy was already 0.938/0.979. Product code did not clear the frozen margin over both explicit sum and difference. Every tested rank exactly matched native PCA-code multiplication; fresh seeds stayed sealed.
- **MA-520:** Learned rank-four shared linear decoding used 12,692 B, but held-out gold log probability fell by 0.607/0.859 nats versus explicit FVs. At the same bytes it differed from native PCA by only +0.000159/−0.000128 nats. Per-vector int8 FVs used 10,188 B, matched top-1 accuracy, and differed from explicit gold log probability by +0.001677/+0.000241 nats. Fresh seeds stayed sealed.

## Interpretation

For the tested aligned functions, static low-rank FV codes are standard shared-subspace compression. Learning the factorization does not add a functional degree beyond native PCA. Function-vector arithmetic in the same coordinates does not itself implement reliable function composition. At the tested rank, int8 quantization is a stronger storage-quality choice than low-rank projection.

These experiments do not challenge the causal usefulness of explicit FVs. MA-516's explicit vectors remain a positive constrained-task signal. They do show that a shared linear basis is not enough to retain that effect at the chosen low rank.

## Family boundary

Do not rerun unchanged linear/PCA function-vector compression or product arithmetic without a different task objective or a representation that is not equivalent to native shared-subspace coding. MA-521 remains a valid distinct test: a demonstration encoder predicts the code directly, compared with explicit ICL and FV extraction. Its direct control must account for encoder parameters, demonstration tokens, code bytes and compile latency. This diagnosis is not a general statement that learned nonlinear or task-aware code extraction cannot work.
