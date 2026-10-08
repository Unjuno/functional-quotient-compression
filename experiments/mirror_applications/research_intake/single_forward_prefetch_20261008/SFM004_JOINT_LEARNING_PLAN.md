# SFM004 — Jointly learned shared trunk + K useful role outputs (standalone design)

**Linked MA:** 1175. **State:** PLANNED / NOT RUN. Research focus: one expensive forward producing genuinely useful **K distinct** output functions; offload/prefetch is a separate SFM006 lane. This is a new experimental protocol proposal, NOT permission to reuse the already opened SFM003 fresh seeds or alter frozen SFM003 thresholds.

## H — Testable marginal claim

Training a single trunk `F_θ` on K distinct supervised task targets, with only a short structured role code `m_j` per task, can retain at least 98% of native one-forward linear-head target performance on independent held-out tasks while reducing **whole inference payload** by >=10% and keeping P95 <=1.10× the fastest byte-feasible native baseline. Test aligned, partly aligned and independent task generators; a favorable aligned-only win is not sufficient. K in {2,4,5,8,16}; increasing K without bound is not assumed beneficial.

## T — Standalone implementation contract

1. **Teacher task worlds:** 16-dimensional inputs with 8-dimensional targets. For each world draw one common nonlinear function and independent private random nonlinear functions. Mix via beta in {0,0.3,0.8,1.0}. A role selects a different target, not merely an output-display coordinate; include tasks with separate order/composition. Never provide the shared teacher intermediate `z` to students.
2. **Trainable architecture:** one 16→64 GELU→8 shared trunk, train from initialization in all conditions. Compare (A) shared+no role, (B) shared+4-angle Givens m per output, (C) shared+diagonal or FiLM, (D) shared+per-role dense 8×8 readout, (E) shared hidden-preactivation+structured nonlinear residual m, (F) native MIMO/MIMMO/NFE when implementable, (G) K independent FFN experts quality reference. Native linear multihead itself executes one trunk exactly once and must not be weakened.
3. **Training:** source-only development seeds 11–13; select LR and width/early stopping before fresh. All candidate baselines receive equal examples, updates, precision, and teacher labels; log training wall and active MAC proxies. Freeze architecture and sample/role identity before separate fresh seeds 201–205, which have **not** been used in SFM003. **Do not run SFM004 until a new immutable JSON protocol with checked source hashes has been committed.**
4. **Measurements:** per-role NMSE and worst-role error, Jacobian/conditional output probe rank, minimum and median pairwise output *functional* separation, input OOD shift, old-task retention, serialized bytes including code+base+decoder+metadata, no repeated-input prefill counted as one forward, actual single heavy-trunk call via module invocation counter, and time P50/P95 across batch/context/hardware. Stress test K=16 for quality/cost collapse.
5. **Full native algorithms:** MIMO/MIMMO/NFE require verified method-specific implementation. If not available, label them BLOCKED while retaining a straightforward shared trunk+linear heads as a **mandatory strong native control**. Always include ordinary pairwise Givens heads exactly equivalent to output-only Mirror: otherwise false novelty.
6. **All-task gradients:** verify `grad_theta mean_j L_j` equals the sum/mean of independent task backward signals on one shared `F_theta` up to float tolerance; sequential/parallel execution semantics are not the cause of functional multiplicity.

## D — Preregistered intended scientific gates

- **PASS mechanism:** true measured one-heavy forward for K=4/5 and 4/5 fresh independent tasks with no output collapse; head transforms significantly improve task loss over no-role output.
- **PASS Mirror-specific (strong):** at beta≥0.3, the Mirror candidate retains useful task accuracy within 2% relative of strongest same-byte shared native while saving >=10% whole inference bytes, <=1.10 P95 and at least one independent quality/byte/runtime axis not replicated by a byte-matched simple alternative. Require new independent task worlds before adoption.
- **FAIL:** ordinary same-byte head/FiLM/native MIMO dominates; noncollapsed output probes fail; off-orbit quality fails 2/5 worlds; or shared trunk is secretly recomputed K times.
- **UNCERTAIN:** no natural task, failed native code reproduction, missing serializer/CPU/GPU calibration, or five-seed intervals inconclusive.

## C — Opposing explanation

If K target functions require different hidden/nonlinear computations and the shared statistic does not preserve the needed information, a cheap output View cannot reconstruct them. A rich native shared-trunk linear head may already span the entire useful output family. Code amortization only helps storage if the extra decoder and code bytes are smaller than ordinary heads and no expensive head-specific nonlinear FFN is repeated.

## U — Uncertainty and unit table

Independent worlds are statistical units, not individual correlated tokens. Report paired per-world differences, task-bootstrap intervals, numerical FP64/FP32 differences, runtime repeat uncertainty separately, and serialize actual file bytes. For uncorrelated quality components `u_c^2 = u_seed^2+u_eval^2+u_num^2`, otherwise add covariance terms; `k_cov=2` is indicative, not a 95% theorem at n=5.

| Symbol | 日本語の意味 | Unit (SI/practical) | Type / domain |
|---|---|---|---|
| x | 入力ベクトル | 1 | real vector R^16 |
| θ | 学習する共有重み | 1 | real weight tensors |
| z | 共有表現 | 1 | real vector R^8, NOT supplied from teacher |
| m_j | task View code | 1 (angles rad) | finite real k-vector |
| K | 論理出力数 | 1 | integer 2/4/5/8/16 |
| beta | private non-orbit proportion | 1 | real in [0,1] |
| L | task error / NLL | 1, nat/token if NLL | finite nonnegative scalar |
| S | 真の保存容量 | byte (practical, not SI) | integer >=0 |
| t | decode latency | s | real >=0 |

Dimensional validation: output m transformations map R^8→R^8; the shared U and D have shapes 16×64 and 64×8. A byte/second ratio is not a quality score; compare these on a constrained Pareto frontier, never sum them.

## Provenance/prior art

- [SFM003 frozen pilot and negative result](SFM003_REPORT.md).
- PA42 native MIMO; PA437 MIMMO; PA438 Network Fission Ensembles; PA16 Parameter Superposition; PA17 BatchEnsemble; PA417 Grouped Value Attention.
- Existing [MA-1175 hypothesis](plans/MA-1175/README.md).
