# MA-1178 Status

2026-10-24 JST — **COMPLETE: exact-representation mechanism PASS; general Mirror-specific / random-start distillation gates FAIL.**

- Experimental `PROTOCOL.json` frozen on research branch before training (SHA256 e279c49bf5e17b16228f05a4f514d43e039440ba2d336154b633b4983b2341bb).
- 20 ordinary teacher MLPs pretrained and frozen (5 seeds x base+3 independent). Three teacher worlds (Mirror-related / FiLM-related / independent). 180/180 student models fitted 750 updates, 180 SHA256 checkpoints sealed before Fresh/OOD generation; 180/180 function-score replays exactly match. 7/7 code tests PASS.
- Oracle source-MLP transfer shows exact three-function representability with Mirror H16: max DEV NMSE 4.717e-14.
- Random-start H16 Mirror has <0.01 Fresh NMSE in 2/5, and beats equal-parameter FiLM + independent by 10% in 3/5 (both predeclared thresholds 4/5 FAIL); generic 3-world FiLM comparison 3/15 vs 8/15 gate FAIL.
- An aligned, synthetic positive H8 result is recorded without promoting overall gate: Mirror has >=10% gains over FiLM and independent students in 5/5 aligned seed pairs; independent tasks favor native students. Mirror CPU eager remains slower than FiLM.
- **Posthoc exploratory (not confirmatory):** 40 additional aligned teacher-core transfer student runs; mirror-family H16 warm start improves median Fresh NMSE 0.010794 -> 0.004165 when the pretrained common core is given to the student.
- No causal language model, attention or GPU-speed claim. No merge to main.
