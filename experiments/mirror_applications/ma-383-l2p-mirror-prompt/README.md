# MA-383 — L2P prompt pool with Mirror prompt generator

Status: protocol frozen before development. Dedicated branch: `research/ma-383-l2p-mirror-prompt-20261009`.

## Mirror insertion

**Mirror insertion:** this experiment adds one Givens angle `m_i` to the shared two-vector prompt basis so that eight retrieved prompt functions can vary without storing eight full prompt vectors.

## H — Hypothesis

A shared prompt basis plus one Mirror angle per prompt will preserve prompt-conditioned output quality at <=60% of the actual explicit prompt-pool and key payload while maintaining the same nearest-key retrieval accuracy.

## T — Frozen design

PA56/L2P retrieves prompt values from input features without task identity at inference. The screen uses eight fixed, paid prompt keys, noisy queries, and a synthetic fixed 8x8 prompt decoder. Compare explicit prompt values, hard tying, scalar gates, generic two-coefficient vectors, and Givens views. Prompt training and retrieval are separated: prompt vectors train from source-task examples; test query retrieval is task-identity-free. All seeds, gates, and storage accounting are in the committed `PROTOCOL.json`.

This is a prompt-pool mechanism test, distinct from the small shared adapter-bank family paused after MA-379/381. No natural continual-learning claim is made.
