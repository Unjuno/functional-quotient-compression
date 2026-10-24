# MA-1177 Status — 2026-10-24 JST

**COMPLETE. Numeric mechanism PASS; exact-copy distillation stability PASS; hard-label learning gain FAIL; Mirror-specific architecture gain FAIL.**

- Protocol file SHA-256 `60277fbc83524c49c44cb1c825a8a1d599a78355b57d7ab97ef80d7a34cd8e08`, uploaded to research branch before any training.
- 72 hard CE + 48 exact-copy soft-target distillation = **120** trained/checkpoint-hash-sealed models.
- Fresh/OOD opened only after the 120-checkpoint SHA-256 manifest was frozen. All 120 checkpoint and evaluation replays matched exactly.
- 8/8 numerical tests; 1 complete hard training and 1 complete distillation re-training gave identical learned tensors.
- Fixed hard CE improves Mirror Fresh CE >=5% in **0/6** pairs at logit scale12 (FAIL). Same-coefficient FiLM comparison shows Mirror-specific improvement **0/6** (FAIL).
- Fixed soft CE eliminates teacher/student drift from exact initial equality in Mirror **6/6** pairs at eps=1e-8 (PASS); FiLM also benefits, so this is numerical correctness, NOT Mirror-specific capacity.
- An additional 48-configuration near-copy perturbation diagnostic is **POSTHOC DEV-only**; it did not prove general distillation improvements and does not alter any preregistered gates.
- This experiment is small synthetic attention classification, NOT true grokking, real natural-language modeling, or modified FlashAttention kernels. Prior historical FAILs intact; no main merge.
