# MA-297 — Mirror codes inside a SETA-style shared sparse subspace

Status: **FAIL (fresh aligned quality gate missed; compute/runtime Pareto also failed)**
Source branch: `research/ma-297-seta-mirror-subspace-20261008`; reconciled branch: `research/ma-297-seta-reconciled-20261008`
Evidence lane: MECHANISM / CONTINUAL / STORAGE / RUNTIME

## H — hypothesis

A per-task Mirror coordinate over discovered shared sparse atoms could preserve aligned task functions with fewer inference bytes than ordinary task-specific coefficients, while unrelated tasks would reveal where private parameters are needed. A Mirror-specific benefit required beating the matched native coefficient control at comparable quality, bytes, and compute.

## T — execution

The candidate was selected randomly from `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-288, MA-292, MA-296, MA-297, MA-299]` using Python `secrets.randbelow(10)`: draw 8 selected MA-297. `git fetch origin --prune` found no MA-297 branch at selection time. This selection is recorded in `PROTOCOL.json`.

A NumPy CPU harness implements one task-0 shared linear map, a fixed correlation-based sparse atom discovery step, and four sequential known task IDs. Development seeds 29701/29702 selected one shared atom pair by mean final held-out MSE over all methods and both families: one/two/four pairs scored 0.661397/0.751343/0.952133. Fresh seeds 29711–29713 used the frozen one-pair setting. Aligned teachers vary through a two-atom view family; independent teachers use unrelated full maps.

Controls were shared-only hard tying, one-angle Mirror, two-free-coefficient same-basis control, rank-1 residual, and independent full task weights. Every fit used only its 64 training examples: a fixed 4097-angle grid for Mirror; least squares for coefficient and independent controls; truncated SVD for rank-1. There were zero gradient optimizer updates. The Mirror state was reconstructed from actual serialized payloads; maximum absolute reconstruction difference was below 1.5e-7. All metadata, IDs, codes, bases, and weights needed by each method were serialized and counted. CPU wall time and inference throughput use the actual coordinate path.

Two audit corrections were required and preregistered post-access: the first timing bypassed the Mirror coordinate, and the first serializer charged unused state to endpoint controls. All initial tables remain as `FRESH_*_INVALIDATED.csv`; a pre-serialization-quality table is also retained. Final corrected results use the same locked seeds and fit setting. Quality is calculated on the serialized-state reconstruction. Neither correction changed the model, fitting rule, or quality metric.

## D — decision

**FAIL** for the registered joint quality/byte/compute claim. The aligned Mirror byte reduction is real, but two of three fresh worlds missed the 1.10× quality gate and the extra coordinate was much more expensive to fit and slower at inference.

| Aligned method | Mean held-out seen-task MSE | Total payload | Increment/task | Fit MAC proxy | Inference examples/s |
|---|---:|---:|---:|---:|---:|
| Shared-only hard tie | 0.07944 | 566 B | 0 B | 0 | 162.2M |
| Mirror, one angle/task | 0.06262 | 704 B | 29 B | 100.7M | 25.6M |
| Same-basis, two free coefficients | 0.05799 | 960 B | 93 B | 49,152 | 62.2M |
| Rank-1 residual | 0.06376 | 962 B | 93 B | 49,152 | 62.5M |
| Independent full weights | ~0 | 2,130 B | 413 B | 36,480 | 163.9M |

Mirror used 0.312× the coefficient control's incremental bytes/task, but its MSE/control ratios were 2.37e8, 1.025, and 1.102 across fresh seeds; the first coefficient result is effectively zero (6.44e-16) and the last ratio exceeds the preregistered 1.10 limit, so only one of three worlds passes. Its CPU inference throughput was 0.43× the coefficient control. The fixed-grid fitting proxy was about 2,048× larger. On unrelated tasks, Mirror mean MSE was 1.644 versus 1.242 for the coefficient control and approximately zero for independent full weights. Hard tying used fewer bytes than Mirror.

## Fact / interpretation / hypothesis

**Facts.** See `FRESH_RAW.csv`, `RESULTS_CORE.csv`, invalidated audit tables, and `VERIFICATION.json`. Corrected payload reconstruction and metric/hash replay pass; final primary metrics are computed from reconstructed float32 payload state. Earlier functions are stored separately and unchanged, so forgetting is zero by construction for this task-ID protocol.

**Interpretation.** A compact angle can reduce task-code bytes for this two-atom sparse screen, but misses the quality gate in one fresh world; its fitting and eager inference costs erase a useful Pareto improvement. Ordinary coefficients are the stronger quality/compute control; hard tying remains smaller. Unrelated tasks require private capacity.

**Hypothesis.** A learned/fused code or a broader sparse basis might reduce runtime or improve task accuracy, but that would require a new experiment and cannot be inferred from this screen.

## C — strongest counter-hypothesis

The synthetic aligned generator is close to a rotation, but the task-0 discovered basis and fixed single-pair coordinate do not span every realized task equally. The near-threshold quality miss may be a basis/support choice; however, adapting that choice to fresh outcomes is prohibited, and the extreme grid-search compute/runtime penalty is independently unfavorable.

## U — unresolved

This is not a full SETA reproduction: no learned routing, optimizer-driven continual updates, language model, natural task stream, or near-convergence capacity frontier was tested. Zero forgetting follows from storing separate task functions and does not establish continual-learning interference resistance.


Integrated from `research/ma-297-seta-mirror-subspace-20261008` at source result commit `e3b74c5`. Four tests and corrected fresh metric/byte/hash replay were rerun on the cumulative branch. Initial timing, endpoint-byte, and pre-serialization-quality tables remain preserved as invalidated diagnostics.
