# MA-795 — Merge/prune overcomplete experts, then recover logical roles

Status: **FAIL** (scoped synthetic screen; Mirror attribution)

The frozen protocol compared independent experts, reconstruction-aware pruning, routing-weighted merging, a small two-expert fit, and a rank-1 residual View with its ordinary native low-rank twin. See [PROTOCOL.json](PROTOCOL.json).

## Prior-art boundary

PA203 motivates using routing statistics to identify and merge redundant experts before compression. PA209 motivates pruning by output reconstruction importance. This screen uses synthetic importance weights and known Gaussian role modes; it does not reproduce MC-SMoE, REAP, or training trajectories.

## Fact

Seed 79501; eight 12×12 linear experts in two clusters; 1,024 train and 1,024 test examples; zero optimizer updates. Independent experts serialize to 5,191 B and achieve weighted test nRMSE 1.05e-7. Prune-only retains four experts at 3,265 B, nRMSE .2508. Merge-only uses two pooled experts at 1,977 B, nRMSE 1.1606. Small-from-start two-expert fit is 1,769 B, nRMSE 1.1011. Merge plus rank-1 View is 3,036 B, nRMSE .02472 (mean per-role .02444, worst role .07882), with 1,152 reconstruction MACs and 240 active-MAC proxy/example. The View exactly aliases its native rank-1 residual control by payload hash `4911749a8d73dcb61b7cb6a669e30a49827a85dbc8fde466172bb5871da7a8c2`; all payload roundtrips reproduce operators exactly. Routing loads are balanced by construction from 128 examples per role; this does not validate a learned router.

## Interpretation

The residual View substantially reduces the error of merging in this synthetic task and saves 41.5% complete bytes versus independent experts. It misses the frozen .02 quality gate, and the same coordinates are ordinary low-rank residual factors. The View also costs more bytes than merge-only/small-from-start, adds residual reconstruction work, and uses the full eight-role router. Thus this is not a Mirror-specific Pareto gain. Pruning alone loses important distinctions under the equal-role test distribution.

## H / T / D / C / U

- **H:** Merge/prune identifies a compact pool and a rank-1 logical View recovers removed expert distinctions at ≤.02 test nRMSE and ≥25% fewer bytes than independent experts, outperforming merge/prune controls without added active compute.
- **T:** Synthetic Gaussian role modes; seed 79501; eight experts, two clusters, dimension 12; 128 train/test examples per role; controls: independent, prune-only, merge-only, small-from-start, Mirror rank-1 residual, native rank-1 residual. Actual compressed NPZ bytes/hash, routed test output metrics, MAC proxies, train/test example counts, single-call CPU wall calibration, zero updates.
- **D:** **FAIL** for Mirror-specific value. Storage improves 41.5% over independent, and error improves over merge/prune controls, but .02472 misses the .02 quality gate and native low-rank exactly aliases the View. The small-from-start/merge controls use fewer bytes and active MACs.
- **C:** The target functions were generated from shared cluster cores plus a rank-1 residual, so ordinary low-rank approximation is the direct explanation. The private perturbations on two roles cause the remaining error.
- **U:** Fresh worlds, learned MoE routing, trained merge/prune procedures, realistic routing distributions, neural/LM loss, training trajectories, GPU latency, and crossover to private residuals.

Fresh stayed sealed after exact native alias and the frozen development quality-gate miss.
