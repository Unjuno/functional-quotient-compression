# MA-643 status

- Status: FAIL
- Branch: `research/ma-643-galore-mirror-subspace-code-20261104`
- Base commit: `19161f6c`
- Protocol frozen before development: yes
- Development complete: yes (64301, 64302)
- Fresh/audit opened: no; frozen failure gate reached

## H / T / D / C / U

**H:** When task gradients reuse a small set of left subspaces, a shared projector dictionary plus task Mirror codes reduces serialized GaLore resume state while retaining fit, relative to one projector per task.

**T:** 16 independent 32x32 task matrices, rank-4 projectors, 64 examples/task and 100 updates/task. K=1/2/4/8 recurring gradient modes, two development seeds. Compared full Adam, standard per-task GaLore SVD projectors, a shared dictionary selected by initial-gradient alignment plus codes, and a native projector-ID control using the same dictionary and decoder. Both inference weights and complete resumable state are serialized. Each condition performs 1,024 task examples and 1,600 optimizer updates. Projector discovery uses 16 SVDs for standard GaLore and K calibration SVDs for shared dictionaries; code selection MACs and wall time are separately recorded.

**D:** FAIL. Shared dictionary and native ID results are exactly identical. At K=1/2/4, full resume payload is 91.9%/92.5%/93.6% of standard GaLore; at K=8 it is 95.8%. Standalone serialized projector dictionary+code payload is 16.7%/22.5%/33.9% of the independently serialized projector payload at K=1/2/4, then 56.8% at K=8. Mean held-out MSE is close, but at K=4 seed 64302 is 1.117x standard GaLore and misses the frozen <=1.10x gate. No Mirror-specific gain over ordinary projector IDs; fresh seeds 64311–64313 stayed sealed.

**C:** Reusing gradient projectors is generic basis sharing. A shared basis changes Adam's coordinate orientation, so even equivalent projector subspaces can change finite-budget MSE. Inference weights dominate the resumable payload and bound the full-checkpoint savings to 4–8%.

**U:** No natural gradient distributions, pretrained language model, or optimizer-throughput benchmark was tested. Fresh behavior is not established. The synthetic setup gives a shared recurrent subspace dictionary from calibration gradients; natural dictionary discovery may cost more.

## Fact / Interpretation / Hypothesis

- **Fact:** 32 development rows (2 seeds x 4 K x 4 methods); actual inference, resume, and geometry NPZ sizes replay; all three tests pass; protocol hash matches freeze; Mirror and native dictionary states/outputs match exactly.
- **Interpretation:** Reusing a small projector bank can reduce optimizer checkpoint bytes in this controlled task, though full-state savings are modest and finite-budget quality depends on basis orientation.
- **Hypothesis:** Similar savings may appear for naturally recurring gradient subspaces, but Mirror-specific utility requires an advantage over standard dictionary IDs or another matched native control.
