# MA-249 — one physical future head + Mirror future-offset views

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `4fe9b5fe51bfcd488ddff9a9702f8cb9bee4b08a`

## Hypothesis

When future-offset teachers share one output matrix under low-dimensional hidden-space Givens transforms, one physical MTP head plus offset-specific Mirror coordinates can recover the independently parameterized heads with lower actual serialized bytes. On unaligned offset heads it should fail or require private residuals.

## Prior art delta

PA09 (Multi-token Prediction) uses a shared trunk with separate future-token prediction heads. The comparison therefore includes ordinary four-head MTP and hard head tying, as well as scalar-gated and rank-1 private-residual controls. This synthetic screen tests only the head layer: a common frozen feature map feeds four categorical future-offset outputs, and students are distilled from either (a) a shared teacher head viewed through per-offset Givens transforms, or (b) independent teacher offset matrices.

## Gates and protocol

See `PROTOCOL.json`. Primary quality is teacher-to-student categorical KL, with top-1 distribution agreement as a second measure. Every full inference payload, including the shared frozen feature map and metadata, is serialized and charged. The Mirror variant rotates a 16D shared hidden vector with four offset-specific Givens views before applying the one shared output matrix.

## Results

Pending development and fresh evaluations.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: pending.
