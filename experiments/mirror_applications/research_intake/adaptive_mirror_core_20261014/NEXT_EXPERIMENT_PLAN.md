# AMA-002 — Learned per-group Mirror use with real physical pruning

**2026-10-14: PLAN ONLY / NOT RUN.** Follow-up to [AMA-001](README.md), which passed limited partial/activation learning mechanism gates but failed native storage and unrelated-task quality plus eager CPU latency. Linked to already registered MA-314, MA-415, MA-550; no new official MA ID.

## Core hypothesis
Do **not** Mirrorize all parameters. Keep ordinary shared FFN as the default. For each task×channel group, learn whether to (a) keep the native shared path, (b) apply a shared low-description Mirror transform with optional learned activation basis, or (c) pay for an explicit private low-rank fallback. Gate/coordinate/activation parameters learn jointly. For actual saved-byte gains, inactive gates and their weights must be physically omitted from a reloadable inference archive.

## Candidate interface
Let `h∈R^d` be shared features; groups `P_g:R^d→R^{d_g}`; shared transform `Q_g(m_tg)∈R^{d_g×d_g}`; learned mirror strength `s_tg`; private strength `p_tg`; shared decoder `U_g:R^{d_g}→R^o`; paid private residual `E_tg:R^{d_g}→R^o`. Then
```
y_t = W_out φ0(h)
      + Σ_g s_tg U_g [ φ_ηg(Q_g(m_tg)P_g h) − φ0(P_g h) ]
      + Σ_g p_tg E_tg(P_g h)
```
with learned sparse/relaxed routing and an explicit `no Mirror` option. Activation coefficients should be shared within groups, not new full per-task MLPs. The final discrete archive must include routing indices, actual stored coefficients, exception terms, matrices/axes and metadata.

## Tests and controls
Run 8 tasks across four families: aligned rotation teachers; partially off-orbit teachers; fully unrelated teachers; and a train-only-PCA16 sklearn digits8 real-image lane. New calibration seed 6000, 3 dev seeds 6101–6103 and sealed new fresh seeds 6201–6205. Do not reuse AMA-001 opened dev or sealed fresh seeds.

Compare: shared-only; native FiLM/IA3 with learned group gates; LoRA rank1/2 and adaptive low-rank allocation; native shared+rank1/private routing; original always-on GELU Mirror; original adaptive activation Mirror; new group-selective fixed-activation Mirror; new group-selective learned activation Mirror; exact equivalent native learned Givens; shared-trunk individual heads. Same training examples, updates, tuning budget, validation-only mask/checkpoint selection.

## Preregisterable gates — not outcomes
H1 partial sharing: test nMSE within +0.01 of best quality-qualified native with worst-task degradation <=0.02; full physical NPZ <=95% of least expensive qualifying native in >=2/3 dev seeds.
H2 unrelated: paid private/native route correctly protects independent tasks to within +0.03 nMSE worst-task of ordinary shared-trunk independent heads. Fail if cheap-looking storage merely hides many full private copies.
H3 orbit: actually use Mirror for useful functions beyond a learned native-group-Givens alias, or explicitly declare no Mirror-specific novelty.
H4 activation: jointly learned activation must outperform fixed GELU and equally-paid activation-only controls on function quality at useful bytes.
H5 runtime: hard-pruned eager and precomputed batch1/64 latency <=1.25× the fastest equal-quality native control and paid runtime RAM reported.
All gates mandatory to open 6201..6205.

## Work order
1. Write fail-first unit tests for no-view identity, orthogonality, autograd flow, exact physical pruning/load, and train-test/source-support firewall.
2. Freeze full source, data and protocol SHA before executing new development seeds in CPU container.
3. Calibrate only on 6000: capacity penalty, L0 gate strength, activation mixture; do not tune on dev/fresh.
4. Evaluate 6101–6103 and strongest native controls; record per-task quality, real whole-bank NPZ bytes, training updates/wall-clock, batch1/64 latency, off-orbit escape and gradient conflict.
5. Open fresh only when ALL preregistered gates hold. Otherwise preserve failures and do not change thresholds.
6. Publish exact runnable source, raw saved model evidence, replay SHA checks, and H/T/D/C/U; keep `main` and canonical MA statuses untouched.

**This file is a next-step experimental protocol sketch, not an executed result or deployment claim.**
