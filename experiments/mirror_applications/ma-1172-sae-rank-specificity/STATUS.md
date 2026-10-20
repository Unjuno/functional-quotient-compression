# MA-1172 — completed verification

- **2026-10-20 JST:** Preregistration on `research/ma-1172-sae-rank-specificity-20261020` was committed before training; source and controls subsequently frozen.
- 30/30 new CPU FP32 trainings, 500 updates each (2 synthetic teacher worlds × 3 seeds × 5 methods), completed. The fresh ID and OOD splits were evaluated only after all 30 dev checkpoint selections.
- **Unit tests:** 5/5 PASS. **Checkpoint packet byte-exact replays:** 30/30 PASS.
- **H1 MECHANISM PASS:** Mirror numerical rank 32/32 vs linear page encoder 8/32 in all six pairs at rel threshold 0.001; FP64 forward recomputation confirms.
- **H3 NON-SPECIFIC:** generic direct-tanh without Householder also reaches numerical rank 32/32 in all six pairs and has higher median energy effective rank than Mirror.
- **H2 MIRROR-SPECIFIC FEATURE-RECOVERY FAIL:** frozen paired quality gate 0/6; Mirror vs direct tanh paired F1 win 2/6; full dense SAE remains substantially better.
- Inference programs are not zero-cost; fixed buffers are deterministic and regenerated, decoder dictionary is independent learned 64×32, and CPU runtime is slower than simple controls.
- Verdict: preserve mechanism as a scoped generic nonlinearity finding, do not adopt Mirror SAE or infer real LLM interpretability gains. MA-1171's earlier FAIL retained.
