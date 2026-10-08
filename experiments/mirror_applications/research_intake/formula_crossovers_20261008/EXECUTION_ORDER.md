# Execution order — standalone formula crossovers

**This is not WORKER_QUEUE.** No automatic claims, no preemption, and no status modifications to the canonical worker baseline. Research branch: `research/mirror-internal-formula-crossovers-20261008`; the original main and MA-255 worker queue remain unchanged. Always consult [formulas and hypotheses](FORMULA_LEDGER.md) and [machine-readable crossover matrix](CROSSOVER_MATRIX.csv).

## Recommended order: cheapest decisive failure first

| Stage | Candidate | Dependency/prior screen | Cheap first gate | Cost-bounded escalation |
|---|---|---|---|---|
| 0 (COMPLETED algebra) | F01..F12 | [frozen Stage0 protocol](FORMULA_STAGE0_PROTOCOL.json) | twelve exact algebra checks, five fresh worlds, all passed | No task-level success or capacity conclusion |
| 1 | MA-1171 | E7 / D63 decoder atom and serializer | exact enumerate 4 blocks x bitwidth/private; verify actual bytes and native equal-budget joint codec | same-base natural LoRA bank then GPU |
| 2 | MA-1172 | MA-691 / KQ-SVD / xKV | coupled attention-score small tensor, diagonal false quality gates, true memory accounting | full tiny LM, long-context native GPU kernels |
| 3 | MA-1174 | RA-Mirror cost direction and J range | 2x2 weighted Jacobian eigen ratio 100, unreachable and incorrect inverse test | task-useful source-only JVP and 10/50/100 updates |
| 4 | MA-1173 | SRM002 vs SRM003 executor and TM001 | independent factorial: signed native vs Mirror; one/two-forward; no oracle intermediate | joint hidden packet valid-path with PTP/AR controls |

**Proposed initial work discipline:** finish Stage 1 dev on CPU first because E7 already has a mathematical counterexample and a stronger non-Mirror joint optimizer; stop early on developmental futility. In Stage 2, do not imply xKV native saving is Mirror saving. In Stage 3, high generalised λ does not mean task value. In Stage 4, supplying a two-forward executor is an extra runtime resource, never zero compute.

## Frozen data contract per candidate

1. Identify whole source task IDs and frozen native base model/optimizer before choosing any view. Distinguish aligned synthetic (mechanism) from independent naturally trained target tasks (capacity).
2. Source/dev seeds 11,12,13 may select code dimension k, task/metric rank tolerance, step count and source learned dictionary.
3. Fresh seeds 101,102,103,104,105 are disjoint whole task identities. No target task oracle delta, audit labels or heldout query Q may tune m, router or private support.
4. Do not selectively stop after favourable fresh outcomes. Freeze all native model baselines, exact physical byte budget and inference kernel before opening fresh. Follow native method first and add m at only one declared interface.
5. Compare native dense/sharing, same-byte simplest linear/FiLM/gate/code, structured m and an independent upper control when meaningful. Report actual files and physical VRAM/P95, model NLL and activation/Hessian scores separately.
6. Include at least one deliberately false counterexample to verify safety: uncoupled G false PASS, wrong upstream KV reuse, shortcut oracle intermediate, gauge-equivalent 'extra model', float bit estimator admitting over-budget payload.
7. Run `python experiments/mirror_applications/research_intake/isolated_geometry_rebased_20261008/selfcheck.py` on a complete local checkout for registry/claims and formula plan integrity; run `python .../pilots/formula_stage0/source/test_formula_audit.py` for a stand-alone algebra smoke test. These commands do not modify the repository.
8. A Stage0 mechanism pass cannot change scientific status UNTESTED; only an active ID-specific frozen PROTOCOL, RESULTS_CORE, VERIFICATION with own experiment branch can. Promotion/merge requires user-directed separate decision and re-allocating IDs against latest canonical worker registry to avoid collisions.

## Measurement table / full units

| variable | 意味 | SI unit / practical | domain and type |
|---|---|---|---|
| m,k | Mirror座標と次元 | 1 | m real k-vector, k positive integer |
| D,L | joint distortion and task loss | 1 and nat/token | finite nonnegative real scalars, fixed denominator |
| S | actual inference storage | byte (non-SI information unit) | integer nonnegative, physical file including headers |
| T | execution latency | s | nonnegative real measured with stated clock/batch |
| J | baseline function Jacobian | 1 | real matrix over fixed witness set |
| r,K | effective local code rank and View count | 1 | positive integers |
| u_c,k_cov,U | quality uncertainty, coverage factor, expanded | units of L, 1, units of L | nonnegative scalars, `U=k_cov u_c` under declared covariance |

**Dimensional check:** only loss terms with the same normalized units can be combined as a quality objective. Serial bytes/bit and seconds must be constrained individually. An inferred view count is not Shannon information gain.

## Cross-domain transfer opportunities

- **Model compression/information theory:** joint paid decoder root + private support + quantized task code (MA-1171).
- **Information geometry/numerical linear algebra:** source-only weighted Jacobian, gauge audit and finite-step reachability (MA-1174).
- **Sequence modeling/control theory:** predicted-state executor and shared packet-mode conditional plans (MA-1173).
- **Systems/memory architecture:** exact lazy canonical cache and coupled KV allocation (MA-1172).

## H/T/D/C/U global quality gate

**H:** At identical native task quality, physical serializer state and latency, a genuinely functional Mirror m offers a measurable marginal improvement over the strongest simple non-Mirror shared/low-rank/matrix method.

**T:** Run the independent, candidate-specific protocol; first algebra+native baseline, then source task/dev and locked five fresh worlds. Minimum n=5 paired whole tasks for screening; >=10 new units for an adoption claim.

**D:** PASS only if the candidate's predetermined per-ID quality/bytes/runtime tolerances all hold and the native same-byte control is strictly worse in at least one actual Pareto axis. FAIL on native control parity, quality/byte overflow or cache/oracle violation; UNCERTAIN for unmeasured native runtime or insufficient witness.

**C:** Prior FQC optimization/coupling or native attention architecture alone may explain gains, with no Mirror-specific value.

**U:** Report u_seed, u_eval, u_num in the same task-quality units and include covariance; with n=5, k_cov=2 yields no guaranteed 95% interval. P95 runtime and serializer bytes have distinct measurement uncertainties. Preserve all negatives.
