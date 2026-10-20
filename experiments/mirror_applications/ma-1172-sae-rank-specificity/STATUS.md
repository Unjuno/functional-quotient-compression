# MA-1172 — status and evidence

- 2026-10-20 JST: Separate research branch `research/ma-1172-sae-rank-specificity-20261020`; README/PROTOCOL/STATUS committed **before training**.
- 30/30 CPU FP32 trainings completed, 500 updates each; independent train/dev/fresh splits. Frozen fresh opened once after all dev selections.
- Tests 5/5 PASS; 30/30 checkpoint packet replay PASS. FP64 forward recomputation confirms rank mechanism. Training+audit reports and hash records preserved.
- **Mechanism H1 PASS:** Mirror rank=32/32 at rel threshold 0.001, 6/6; matched affine rank=8/32, 6/6.
- **Specificity H3 FAIL as unique mechanism:** same-address direct-tanh without Householder reaches rank=32/32, 6/6; generic cubic 28–32.
- **Quality H2 FAIL:** strict pair pass 0/6. Mirror ranks do not translate into superior feature recovery. Strong direct dense SAE much better.
- No claim of native LLM SAE interpretability, convergence-limited capacity, free virtual parameters or dominance over simpler nonlinear map. MA-1171 historical FAIL not modified.
