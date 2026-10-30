# MA-1183 — Shared Nonlinear Filter / First-Principles Mirror Experiment

- Frozen before main train: 2026-10-30. No independent confirmations yet.
- Question: can an unknown shared nonlinear kernel + 3 learned Householder codes jointly learn related functions from task-labelled input/output data more accurately than same physical parameter FiLM and a strictly encompassing coupled rank-1 model?
- Primary partially aligned task: 65% oriented common nonlinear function, 35% unrelated private function, analytic labels but teacher weights/codes withheld.
- Positive aligned-only and unrelated negative worlds are diagnostic; use ReLU and SiLU separately. One View calculated per input (explicit task ID); no MoE, no router or loop in this stage.
- 7 baselines x 3 worlds x 2 activations x 5 distinct seeds = 210 students. Independent model h5 is approximately matched parameter budget, independent h12 is capacity upper reference.
- Pilot ran before prereg on held-out synthetic seed 707 only, with 42 conditions x 350 steps x 512 training samples per task; never include in L1 conclusions.
- Primary gate: ReLU 35%-private world, at least 4/5 seeds with Mirror Fresh NMSE <=0.90 of all FiLM, coupled-rank1, parameter-matched independent. No tuning using Fresh or OOD. If FAIL stop without new 10-seed confirmation.
- Every checkpoint SHA256 sealed before any Fresh/OOD construction, source and protocol hashed and unmodified.
- One-view-at-a-time conditional computation with explicit condition ID: conditional inference does not establish self-routing/no-oracle LM behavior.
- Run: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python source/run.py train`; then `... score`; then `... verify`.
- Hardware: CPU eager PyTorch 2.10.0+cpu, float32, one compute thread; do not claim GPU parity.
