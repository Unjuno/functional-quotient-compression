# A10b exploratory follow-up — Does threefold optimization work remove the Mirror deficit?

Locked before any A10b training. **Designed after observing the A10 main audit**; not independent preregistration of the overall Mirror research hypothesis.

- New synthetic worlds 906–909, 2 paired seeds 0/1, single FP32 serialized cap 1024B, 3 fixed methods `mirror`, `shift`, `film`.
- Two regimes with identical learning-rate/batch/data generation/initialization: short=1000 shared scaffold updates + 300 held-out code-adaptation updates; long=3000 shared scaffold updates + 900 held-out code-adaptation updates. Batch128, AdamW lr0.01, no early stopping.
- The model initializer receives world ID, method, cap and seed only; no teacher thresholds. Meta train uses old 16 task labels; adaptation sees only 8 new tasks × 32 support pairs; IID/OOD never accessed before locking model files.
- 4 worlds × 2 seeds × 3 methods × 2 regimes = **48 additional models**, six evaluation metrics per run. Compare per-world NRMSE changes within method, and same-regime Mirror vs Shift/FiLM. Do not assert matched FLOPs across different model architectures.
- Quality: known-task dev and new-task IID/OOD mean NRMSE <=0.20. PASS for optimizer-only rescue only if at least three of four worlds that fail short Mirror pass long Mirror; report other methods and any resource increase. Otherwise no rescue demonstrated. No compression claim even if Mirror improves.
- Limitations: four synthetic worlds from the same ramp family, two seeds/world, test designed after A10 results, 3× optimizer work not priced into weight bytes. Report all failures and run wall-time costs.
