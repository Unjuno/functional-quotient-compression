# MA-274 — BOFT logical expert views

## H — Hypothesis

A shared physical expert plus compact orthogonal/butterfly Views may provide useful routed expert diversity at lower actual bytes than untied MoE and outperform simple gates/rank-one controls.

## T — Conditions

Four-domain synthetic regression, fixed 16-dimensional features, 4 logical experts, 2,048 train and 1,024 audit examples. Fresh worlds 27410–27412 × seeds 0–2; 63 rows. Shared router trained/evaluated separately; minimum audit accuracy was 0.9971. Compared untied MoE, hard tying, IA3 gates, rank-one residual, one-plane Givens, four-stage Givens/BOFT view, and independent expert upper. Fresh protocol/source frozen at `ee4d3a90` before fresh. Canonical payload includes expert tensors, router, all view codes and metadata. CPU only.

## D — FAIL for BOFT-specific gate; structured shared-view result is scoped

| Method | Routed NRMSE | Oracle-route NRMSE | Router accuracy | Payload B | Fit seconds |
|---|---:|---:|---:|---:|---:|
| standard_moe | 0.009194 | 0.001720 | 0.99816 | 4,260 | 0.305 |
| tied | 0.215677 | 0.215677 | 0.99816 | 1,178 | 0.313 |
| ia3_gate | 0.171601 | 0.171507 | 0.99816 | 1,449 | 0.339 |
| rank1 | 0.203021 | 0.202954 | 0.99816 | 1,279 | 0.339 |
| givens_mirror | 0.008678 | 0.000401 | 0.99816 | 1,218 | 0.454 |
| boft_mirror | 0.008788 | 0.001280 | 0.99816 | 1,716 | 4.158 |
| independent_upper | 0.008479 | 0.000000 | 0.99816 | 4,265 | 0.000 |

Fact: fresh four-stage BOFT view reached routed NRMSE 0.00879 at 1,716 B, compared with untied standard MoE 0.00919 at 4,260 B (about 60% lower payload). One-plane Givens reached 0.00868 at 1,218 B and trained in 0.454 s; BOFT trained in 4.158 s. Givens therefore matched or slightly exceeded quality with 29% fewer bytes and ~9.2× lower fit time. IA3 and rank-one controls had much worse errors (~0.172 and ~0.203); hard tying was ~0.216.

Interpretation: one shared physical expert can support four logical functions in this deliberately aligned task, with a large storage reduction versus untied experts. The four-stage butterfly did not add value beyond the cheapest Givens view and failed the registered requirement to beat a simple control by ≥10%. This is a structured shared-view signal, not a BOFT-specific advantage or general expert capacity claim.

Hypothesis: expert-view complexity should be selected by empirical task-orbit dimension; extra butterfly stages help only when Givens residuals show structured errors.

## C — Strongest counter-hypothesis

The teacher is specifically a shared base matrix followed by a two-channel rotation, so the Givens control exactly matches the generated geometry. The four-stage BOFT has more parameters but is not needed for this task. The task is synthetic, low-dimensional, and uses a separately supplied task router.

## U — Unknown

No natural language or vision MoE benchmark, scaling of expert count/dimension, unconstrained learned router, load-balance stress test, GPU runtime, or quality frontier over private residual budgets was measured.
