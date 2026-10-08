# MA-401 — FiLM versus Mirror feature conditioning

Status: **PROTOCOL FROZEN BEFORE DEVELOPMENT**

## H — hypothesis

A four-angle context-specific Givens feature View improves four-context digit classification over a shared MLP at lower actual bytes than FiLM, and beats same-dimensional grouped FiLM/rank-1 controls.

## Prior art

PA63/PA106 establish FiLM as a cheap conditional feature affine transform. The Mirror-specific claim is limited to its marginal improvement at actual byte and active-compute budgets.

## T — frozen protocol

Use sklearn digits v1.8.0 with a fixed raw-array hash; create four deterministic pixel permutations with labels preserved. Each world has one stratified 60/20/20 split reused across contexts. Train one shared 64-64-10 MLP, fit context codes on train only, compare no code, four-angle Givens, grouped scale FiLM (4 values), grouped affine FiLM (8 values), rank-1 output adapter, and four independent MLPs. Development worlds 40100/40101 select LR from {0.003,0.01}; fresh worlds 40110/40111/40112 remain sealed. Complete serialized inference-bank bytes are authoritative.

Gates and budgets are frozen in `PROTOCOL.json`.
