# MA-1167 Status — 2026-10-20 JST

**SCOPED COMPLETE.** Source and full raw checkpoints available in the accompanying reproducibility archive and hashed in VERIFICATION.json.

- H1 synchronous all-to-all virtual-Mirror mechanism: **PASS**, 5 nodes ×5 source nodes ×3 Mirror views =75 candidate logical MLPs per Loop; 289 trainable scalar coefficients in one model tensor; 7/7 unit tests.
- H2 Mirror-specific quality at fixed 289 learned coefficients: **FAIL**. Mirror T3 loses to affine T3 and native residual MLP in **6/6 paired fresh trials each**. T3 beats Mirror T1 in 4/6.
- H3 same-model CPU batching: **PASS**. Batch1 881us vs sequential-edge 15,597us. Batch64 2,414us vs 18,556us (3-seed medians). **Native MLP remains faster.**
- Evidence: **24 trained checkpoints**, all fresh/OOD and model packet replays 24/24 PASS; 12 full-T gradient VJP diagnostic decompositions matched, with no negative core-gradient pairs on held-out 8-sample local checks. CPU only.
- Accounted storage: trainable 289×FP32=1156B; model packet with 20B header=1176B; current fixed seeded decoder runtime buffers 293480B; generated virtual weight workspace 33000B; source/runtime library separate.
- **No pruning. No inferred GPU result, Sparse Attention, global optimality or learnable arbitrary physical-to-virtual address mapping.**
- Prior negative results preserved; main untouched.
- [REPORT_JA.md](REPORT_JA.md), [VERIFICATION.json](VERIFICATION.json), [results/RESULTS_BRIEF.csv](results/RESULTS_BRIEF.csv), [PROTOCOL.json](PROTOCOL.json).
