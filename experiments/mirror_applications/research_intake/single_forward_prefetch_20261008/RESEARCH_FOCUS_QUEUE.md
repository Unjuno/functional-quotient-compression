# PRIMARY RESEARCH FOCUS — Single Forward → Many Useful Outputs

Date: 2026-10-08 JST. Branch: `research/mirror-single-forward-prefetch-20261008`. **This is a research-only prioritization, NOT the worker's WORKER_QUEUE.** Live worker MA-255 and its baseline/scientific status are untouched. All new study plans stay isolated, are provisional, and need ID reconciliation if promoted later.

## First principle

Given one expensive computation `z=F_θ(x)`, can a small code `m_j` and cheap `T_{m_j}` yield **K=2/4/5/8/16 useful, different task outputs** `y_j=T_{m_j}(z)` with better task-quality / *incremental true bytes* / active computation frontier than native shared-trunk multiple heads, MIMO/MIMMO, BatchEnsemble, NFE, and independent experts?

A single forward from a heavy trunk is **not novel in isolation**: PA42, PA437, PA438 and a conventional shared backbone with K heads already do this. The measurable hypothesis is **cheaper task-specific functional multiplicity** through a low-description geometric code, at comparable task utility and runtime. Offload/prefetch is a separate *secondary* application, not the flagship claim.

## Ordered independent research queue

| Rank | ID | Action | Gate before moving |
|---|---|---|---|
| **1** | **MA-1175 / SFM003** | Audit learned short Givens codes from one frozen shared trunk, K4/K5, favorably aligned and off-orbit synthetic tasks | Completed **synthetic stage only**: beta0 exact; beta0.8 FAIL; native equivalent Givens M0; all raw files in repo |
| **2** | **MA-1175 / SFM004** | Train shared trunk + K codes *end-to-end*, no oracle/source-known latent; K2/4/5/8/16; hold out task families | Useful function quality, noncollapse, one real heavy call, bytes, P95 vs ordinary shared multi-head/native; [protocol](SFM004_JOINT_LEARNING_PLAN.md) |
| **3** | **MA-1175 / natural** | Repeat on real classification/adapter-trained specialist families; frozen matched base/teacher provenance | Separate independent tasks, no target trained delta leakage; target NLL/task accuracy and full serial cost |
| 4 | MA-1171 | Exact codec × shared basis × paid code × private exception, at actual byte budget | Pareto vs non-Mirror joint allocator, no false saving |
| 5 | MA-1172 | Coupled function-sensitive KV/head allocation; reduce repeated states only when useful | Real attention scores & NLL, INT4/KQ-SVD strong native controls |
| 6 | MA-1176 | Physical group prefetch & CUDA overlap **after** 1175 task utility | Real GPU bandwidth, TTFT/TPOT, block transfer/eviction, SpecMD/Pre-gated native baseline |

MA-1173/1174 and other previously registered experiments stay available; this focus does not cancel them or override the live worker queue.

## What SFM003 established (not an adoption claim)

- On 5 fresh synthetic worlds each at K4 and K5 with deliberately output-rotated teacher tasks, **short Givens codes trained from labeled outputs** reproduced heldout task outputs to approximately numerical precision.
- For K5, one heavy call with 5 learned output readouts measured ~3.87x faster than **deliberately** recalculating the same heavy trunk 5 times, not faster than a strong one-forward multihead native.
- K5 output Mirror saved measured NPZ payload (6958 B vs 8556 B for full linear heads) but took *more* CPU inference time (median ~0.0675 ms vs 0.0485 ms) and lost accuracy when independently nonlinear task components were included.
- **Plain block-Givens heads have exactly the same function class and FLOPs as this Mirror output transform**, meaning scientific Mirror-specific uniqueness is M0 at this stage.
- The one-pass idea remains a serious experimental research target, provided it shows useful task output diversity and material resource benefit against native equivalents.

## Required scientific controls and claim hygiene

Measure both ① per-task quality/retention and ② non-redundant output function diversity; multiple different coordinate representations do not count as independent capabilities. Count true base, heads/codes, per-task private state, router, tokenizer/metadata, active MACs and physical GPU memory. Compare at fixed training updates *and* near-convergence, without pretending the first is capacity. For actual latency, include the strongest native one-forward baseline, plus folded experts when resident RAM allows. No GPU claim from CPU eager, no native paper reproduction claim without verified source/checkpoint. Use dev seeds 11–13, fresh 101–105 in SFM003; future experiments predeclare disjoint new fresh seeds and freeze before evaluation.

**Stop/adopt:** A favorable synthetic orbit is feasibility evidence. ADOPTED only after genuine natural output tasks, independent (>10 task/seed units) replication, same-byte native comparison and real inference timing.
