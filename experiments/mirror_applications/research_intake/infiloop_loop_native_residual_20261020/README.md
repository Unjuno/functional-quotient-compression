# InfiLoop prior-art intake — 2026-10-20

**Source:** [Pengxiang Li et al., *Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals*, arXiv:2610.11570](https://arxiv.org/abs/2610.11570), released 2026-10-08 (X announcement [2026-10-19 06:57 UTC](https://x.com/pixeli99/status/2112075683954659684)); [official code](https://github.com/pixeli99/InfiLoop).

**Evidence status:** external paper figures, independently inspected primary arXiv text and official GitHub README, **NOT** an independent reproduction or new MA result. This intake does not change earlier MA verdicts or allocate an experiment ID.

## Mechanism (paper, mathematically exact)
A shared 7M-parameter recursive Transformer uses a loop output candidate z_t = f_theta(h_(t-1), x). Instead of carry-last h_t=z_t, each candidate gets score s_t=q^T RMSNorm(z_t), content mass e_t=exp(s_t) and common decay 0<beta<1. Accumulate:

N_t = beta N_(t-1) + e_t z_t
D_t = beta D_(t-1) + e_t
h_t = N_t / D_t
gamma_t = e_t/(beta D_(t-1)+e_t)
h_t = h_(t-1) + gamma_t (z_t-h_(t-1)).

All values/scalars dimensionless and N,h,z in R^d. This is exact normalized content-weighted loop-output aggregation with O(d) persistent recurrent state and O(d) arithmetic per iteration, **not** an O(d)-activation-memory bound for ordinary full BPTT. q has d learned scalar elements, beta one more per recurrent axis; paper uses independent inner/outer scorers and decays.

## Evidence in paper, context and qualifications
- Sudoku-Extreme 7M total test result: 97.9% exact accuracy vs FPRM 94.2% and paper-reported TRM 74.7%. ARC-AGI-2 pass@2: 13.6% vs FPRM 6.2% and TRM 7.8%. These are under paper benchmark protocols, not our replication.
- At *matched long test-time depth* of 1,872 executed layers, InfiLoop 90.9% vs TRM 58.1%; TRM peaked at 62.2% at 468 layers. Different evaluation slice from final 97.9% table.
- In loop-depth scaling to 24,960 executed layers, InfiLoop 92.74%, FPRM 91.63%, TRM 83.65%; paper claims continuing improvement past 20,000.
- An explanatory interpretation supported by analysis: noisy loop updates can overwrite correct predictions, and an adaptive update can preserve earlier states. Ablations: both inner and outer axes outperform only one, and content scorer is useful beyond fixed temporal decay.
- GitHub README (as checked 2026-10-20) publishes Sudoku code without pretrained weights; code for the other benchmark tasks was announced as future work. Independently measured performance not yet available.

## Relationship to Mirror project (not the same novelty)
**Overlap:** shared physical backbone across loops; learnable acceptance/retention of candidate states instead of unconditional recurrent overwriting; fixed-size recurrent accumulation instead of storing all older loop outputs at inference.

**Differences:** InfiLoop does NOT generate many virtual MLP weight tensors from a Mirror decoder, does NOT learn many-to-many physical-to-virtual weight addresses, does NOT enumerate or prune all skip/pointer connections, and does NOT distill a loop-heavy train-time supergraph into a small one-pass inference model. It assumes recurrent test-time computation is worthwhile; Mirror project's current target is train-time large logical structure and minimized compiled inference bytes.

**Potential integration:** use the exact recurrence as a stable state aggregation primitive *during supergraph training*, after placing candidate states in a canonical space if views rotate activation axes. The orthogonal reflection Q has inverse Q^T; any intervening tanh/ReLU remains noninvertible. Content-score gamma_t is a *loop-state acceptance score*, **not a proven edge-pruning or physical-parameter saliency signal**.

## Proposed falsification protocol (unrun)
- H: content-weighted loop summaries reduce degradation/gradient-conflict of the no-prune full virtual Mirror supergraph and/or make trained route significance more identifiable, at equal physical learned coefficients and matched runtime.
- T: Pair full virtual-Mirror (63 candidate edge-views, all active) and same-budget direct affine shared-basis with three matched recurrence variants: carry-last; learned scalar damping / fixed-beta EMA; InfiLoop pseudoquery+learned beta. Count learned q/beta against shared physical scalar budget in **all** methods. Use at least 3 independent seeds, aligned & nonaligned teachers plus algorithmic state-retention task; track train/fresh curves versus loop depth, candidate overwrite events, single-edge causal ablations, gradient cosine, actual inference bytes/runtime/active MAC.
- D: mechanism supported only if state retention and fresh task quality improve over simpler damped controls; Mirror-specific utility only if Mirror+InfiLoop wins byte-and-compute-matched non-Mirror control. Evaluate whether gamma_t rankings predict post-retraining deletion loss; reject it as pruning signal when rank correlation/calibration is weak.
- C: an identity/no-op candidate may receive high score without useful progress; state convexity limits certain representational updates; q/beta cost and repeated computation can dominate; early score suppression may starve useful training routes.
- U: whether gains transfer beyond Sudoku puzzle tasks to free-form learned Mirror graph, how to train through long loops efficiently without assuming O(d) BPTT memory, whether final inference-time loop removal retains trained function.

**No experiment has been started, no additional claims made.**
