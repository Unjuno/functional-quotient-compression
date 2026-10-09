# MA-556 — Conditional modulation with sparse private residual

Status: **FAIL for registered storage frontier**  
Branch: `research/ma-556-conditional-private-residual-20261009`  
Base: `e5916d18`  
Prior art: PA106 FiLM; PA18 LoRA/private adaptation

## H

A shared conditional basis plus a small sparse per-function residual can retain useful quality as task heterogeneity grows, locating the point where shared View codes need private capacity.

## Frozen screen

Construct 8 logical 8×8 operators with a shared rank-2 task-conditioned component plus controlled private noise at heterogeneity levels 0, .1, .3, .6. Compare shared base only, rank-2 Mirror basis, Mirror plus top-k private residual (k/N = 0, .05, .1, .25, .5, 1), and independent operators. A byte-matched direct coefficient basis is included. Two seeds; oracle SVD fit is a mechanism/storage screen with zero optimizer updates. Serialize every basis, code, residual, index and metadata; report held-out operator nMSE, bytes, multiply-add proxy and wall time.

PASS for the shared/private frontier requires a residual fraction below .25 to reach nMSE ≤1e-3 at heterogeneity ≤.3 and at least 20% fewer bytes than independent storage. FAIL for Mirror-specific claim if direct coefficients match; the main outcome can still establish the residual escalation frontier. No fresh worlds opened after the Mirror/direct gate.

## H / T / D / C / U

- **H:** Sparse private state repairs shared conditional views at predictable heterogeneity thresholds.
- **T:** Eight 8×8 operators, heterogeneity 0/.1/.3/.6, two dev seeds, six residual fractions, 80 rows; oracle SVD, zero updates.
- **D:** FAIL for storage frontier / Mirror-specific value. At rho=.1, 10% residual reached mean nMSE .00086 at 2548B vs independent 2539B. At rho=.3, 25% was borderline (.00098 mean; one seed >1e-3) and 50% reached .00015 at 3778B. At rho=.6, 50% reached .00095 at 3778B. Direct coefficients match Mirror.
- **C:** Ordinary shared low-rank basis plus sparse residual is a direct native control.
- **U:** Learned residual selection, actual FiLM/LoRA fine-tuning, natural tasks, serving latency.
