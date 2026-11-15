# MA-1013 — SSDU scan-level Mirror code

This synthetic screen tests a strict split-k-space constraint: adapt only a compact scan code from measured training samples while freezing the prior; score on disjoint acquired samples and use analytic fully sampled truth only after adaptation. Ordinary affine, FiLM and full-private fine-tuning controls are included.

PA307's SSDU mask-splitting principle motivates the leakage audit. This is an independently implemented synthetic analogue, not an acquired-data or VarNet reproduction. Frozen protocol: `PROTOCOL.json`.
