# MA-1166 Phase F — Learnable many-to-many physical↔virtual correspondence (PROPOSAL)

**2026-10-20 JST. NOT FROZEN, NOT RUN.** This is an experiment design proposal after the verified summary of Phase E. All Phase A–E frozen results, source hashes, and historical statuses remain unchanged.

> **Mirror insertion:** this experiment adds learned low-description correspondence coordinates `m_e` to the selection and readout of a common physical virtual-weight generator for arbitrary candidate graph edges (including long skips and loops), so that multiple useful edge functions can share the same physical coefficients without the fixed-seed address bottleneck.

## The exact failure to address
Phase E's one trainable 442-scalar tensor is split into **64 shared core**, **126 edge gates**, and **252 per-edge view codes**. Despite 126×110 = 13,860 generated logical weights, physical virtual addresses are fixed by deterministic seed. The direct affine control is better in fresh error on 5/6 paired conditions; native442 is better on all 6. Three sync iterations cost 378 virtual MLP evaluations versus 126 serial; no compute-neutral superiority exists.

Phase C's oracle proof copied *both* teacher shared core and code and fixed teacher edge logits. It is only a representability check, **not** evidence that teacher connection IDs or Mirror directions can be learned by backprop.

## Research question

If virtual weight addressing/physical coordinate grouping is learned along with the single core and pointer-connection strengths, does an initially fully enumerated virtual supergraph discover reusable functional structure that fixed addressing and matched low-rank/native controls fail to discover? Can it compile to a small one-pass or sparse standard runtime without retaining all synthetic Mirror paths?

**Experimental novelty is the learned mapping**, not multiplying logical weight counts or adding another dense Mirror/Loop sweep.

## Proposed differentiable mapping architecture (illustrative, NOT frozen)

For candidate directed edge e=(i,j,t,r), physical coefficient object u, and B latent bank slots, define

- `a_(e,b) = softmax_b((s_i + t_j + l_t + v_r)^T k_b / τ)`.
- `z_e = Σ_b a_(e,b) P_(e,b) E_b(u_core)` (P may be deterministic inverse-addressable signed permutation/Householder).
- `w_e = z_e + ρ Q_e^T tanh(Q_e z_e + β_e)`, then reshape into native 6→8→6 virtual MLP weights.
- `y_(j,t+1) = y_(j,t) + Σ_(i,r) g_(i,j,t,r) MLP(y_(i,t);w_e)`, all eligible edge outputs computed as one batch per synchronous step.

Here i/j are source/destination node IDs, t loop step, r view ID, a learned normalized address distribution, s/t/l/v small shared lookup codes (source, destination, depth, view), k_b fixed or counted slot keys, E_b maps different shared core subsets into virtual-weight atom vectors, P is a reversible address permutation/orthogonal Mirror view, g a soft connection gate, Q_e a counted-code mirror, and β_e a counted offset. All feature coefficients SI dimensionless; E_b and P_(e,b) must output vectors of matching virtual-weight dimension N, and the normalized sum and Q^T tanh return R^N. `P^T P=I` is a coordinate inverse only; the nonlinear weight transformation is not globally invertible.

**Avoid a free dense E×P address table.** Prefer factorized per-node/per-loop/per-view codes, B=4/8 and sparse top-k readout; count all embeddings, slot keys, generated address logits, code generators, binary indices, program seeds and any routing buffers in inference payload. Fit into the same **442 learned scalar total** as Phase E by reallocating per-edge code/gate budget to real shared core and structured codes. The outer runtime should avoid full E×B×N materialization when possible.

## Two-stage falsification gate, no early pruning

### F0 — address discovery audit (oracle-support versus unknown-support)

- Use Phase E's fixed teacher architecture but entirely **new independently drawn teacher seeds** and samples. The oracle-known-support arm reveals **only which edge IDs exist** (not teacher weights, shared core, codes or intermediate activations), while the unknown-support arm learns all edge probabilities.
- Keep all edge computations available; no hard removal during learning. Expose unknown support only via train input/output task loss.
- Check learned support recall/precision, causal deletion–retraining influence, address-code entropy, marginal feature reuse (not just shared weight overlaps), and optimization diagnostics `∂L/∂u_core`, `∂L/∂address`, `∂L/∂gate`; group norms normalized by the number of coefficients per group.
- **Stop/hold** if the oracle-known-support learnable model cannot train to within the same error tolerance as the oracle-aware affine/shared-basis control after equal optimizer/update budget. This isolates a decoder/parameterization failure before expensive combinatorial structure search.

### F1 — naturalizable structural discovery

- World A: same-family aligned teacher with independently generated weights; World B: independently weighted dense/sparse edge teacher; World C: compositional/noncommuting function teacher with unseen combinations of allowed long-skip paths. Freeze whole teacher/task IDs across train/dev/fresh.
- Paired fixed total `P=442` physical learned scalar budget in all shared variants. Perform a development-only architecture/B/range/temperature and optimizer study; freeze budgets, seeds, train/dev/fresh and hyperparameters BEFORE fresh generation/evaluation. For statistical generalization consider >=5 independent teacher seeds per world (not reuse Phase E 461/463/467 as fresh confirmations).
- Baselines (all at actual price): 1) Phase E fixed-address Mirror; 2) learned low-rank/affine address assignment; 3) learned FiLM/gating routing; 4) native dense MLP with 442 params; 5) unrestricted independently parameterized edge MLP teacher/capacity upper bound (report different storage); 6) address-map shuffled and oracle-coded ablations.
- First evaluate **without any route/weight pruning**. Only after measuring quality and gradients use a separate experimental stage with gradually annealed connection gates, dev-selected final deletion thresholds, causal edge ablation, short refit, and final compilation.
- Compile two options: (A) fully expanded ordinary sparse MLP/graph, (B) retained shared coded virtual weights + sparse pointer metadata. Report actual model serialized B, loaded RAM/peak workspace, #effective MAC, measured eager CPU time, compiled/fused CPU and GPU timing when available. Avoid training-only compute savings as deployment claims.

### F2 — sync/serial control

Do not compare SYNC3's 378 edge-view evaluations to SERIAL's 126 and claim parallel compute savings. Compare **matched edge evaluation/MAC** (e.g. 3 successive serial traversals, 3 synchronous stages, and a single cached shared projection with cheap views), report critical path and synchronization separately, and measure GPU only where hardware is available.

## Decisions

- **Mapping mechanism PASS:** the learned mapping changes output behavior and successful address/connection discovery is repeatable with causal code swaps, not an exact coordinate gauge.
- **Mirror specificity PASS:** fresh loss improves >=5% relative to the **best** matched learned affine/FiLM/low-rank direct mapping across >=4/5 held-out teachers *at equal learned scalar count, actual stored bytes and active MAC*; otherwise FAIL/UNCERTAIN.
- **Deployment compression PASS:** a final compiled model retains dev-selected quality within +5% relative fresh task loss, and Pareto-dominates the best matched byte/compute ordinary/shared control in serialized B **and** execution cost (unless a preregistered single-axis Pareto trade is explicitly selected).
- **Stop:** oracle-support F0 not trainable, or mapping metadata bytes exceed independent native weights, or false prune signals persist after causal refit. Do not jump to larger supergraphs.

## H / T / D / C / U

H: learned physical-to-virtual mapping resolves fixed-address optimization bottleneck while providing actual compact implementation.
T: oracle-index-only F0, unknown-index F1, multiple held-out teacher worlds, development-only tuning, paired controls and checkpoint replay.
D: above staged pass gates.
C: gauge symmetries, nonidentifiability, code overhead, low-rank control equivalence, gradient conflict, different training objectives under sampled compute, upstream expert-identity leakage.
U: convergence at matched training compute, address identifiability, hardware runtimes, external validity beyond synthetic small models.

**No new experiment results or candidate adoption are claimed here.**
