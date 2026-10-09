# MA-530 — Top-k SAE feature logical experts

Status: **FAIL**. Branch: `research/ma-530-sae-feature-experts-20261009`. Frozen protocol hash: `b2fd285bc2c905e9f9da74fc9de1a6c4214fae0d60b6af97677f1b249124c7ae`. Prior art PA102 studies SAE feature steering and feature-selection side effects.

## H — Hypothesis

A support-trained query-only router selecting among 16 shared-pool SAE feature experts can approach a learned-router explicit-FV expert bank within .10 gold-logprob nats and .05 accuracy at no more than .60x total router-plus-expert bytes, while using no more bytes than global per-expert SAE OMP16 lists.

## T — Frozen execution

Pinned Pythia-70m and the pinned layer-3 SAE used for MA-526–528. Each seed supplied eight support examples and eight disjoint query examples for each of 16 relation tasks. One affine 512→16 softmax router was trained for exactly 1,000 Adam updates on support-source hidden states only; features were standardized from support data and normalization state was charged. At test time its argmax selected a task expert using only the query's source input. Expert controls were oracle-routed explicit FVs, learned-router explicit FVs, learned-router global OMP16 lists, and the shared residual-selected pool64 OMP16 View. No intervention and one shared mean FV were included.

The candidate payload includes router weights, bias, normalization, codes, feature indices and metadata. Every learned-routed control pays the same router state. The standalone SAE is fully charged. Fresh seeds 53011–53013 were not opened because routing and quality gates failed.

## D — Decision

**FAIL; fresh data remains sealed.** The query-only router predicts task identity at 0.477/0.516 accuracy (macro-F1 .479/.512), below the .75 gate. The Mirror expert bank also misses same-router explicit-FV quality and is 150 B larger than global OMP16.

| Seed | Method | Total payload B | Accuracy | Gold log-prob | Route accuracy |
|---|---|---:|---:|---:|---:|
| 53001 | Oracle-routed explicit FV | 34,214 | .6172 | −6.625419 | 1.000 |
| 53001 | Learned-router explicit FV | 72,772 | .6172 | −7.219722 | .4766 |
| 53001 | Learned-router global OMP16 | 41,786 | .4531 | −8.130589 | .4766 |
| 53001 | Learned-router shared pool64 OMP16 | 41,936 | .5000 | −8.232373 | .4766 |
| 53002 | Oracle-routed explicit FV | 34,214 | .5547 | −6.651887 | 1.000 |
| 53002 | Learned-router explicit FV | 72,772 | .5547 | −7.348923 | .5156 |
| 53002 | Learned-router global OMP16 | 41,786 | .4766 | −8.311271 | .5156 |
| 53002 | Learned-router shared pool64 OMP16 | 41,936 | .4844 | −8.510637 | .5156 |

The candidate bank is .576x the learned-router FV payload, meeting that byte ratio, but misses the route and expert-quality gates: relative to learned-router explicit FVs it loses 1.013/1.162 nats and .117/.070 accuracy. It loses .102/.199 nats to learned-router global OMP16 and costs 150 B more. Including the required 4,204,391 B SAE, standalone Pythia+SAE+Mirror deployment is **4,173,555 B larger** than Pythia plus the learned-router FV expert bank.

Router fitting took 3.945/4.463 s and 1,000 updates; query routing took 2.913/2.860 s for 128 examples. The shared expert path total wall time including support extraction, router fitting/routing, sparse coding and candidate evaluation was 25.47/25.70 s; global OMP path was 28.68/28.71 s. These are unoptimized CPU measurements. Router and feature-code construction operation proxies are in each `metrics.json`; actual serialized totals are authoritative for storage.

## Evidence classes

**Facts:** Four tests pass. Deterministic replay matched ten paid payload hashes, core metrics, splits, router predictions and selected pool exactly. Fresh seeds remained locked.

**Interpretation:** The router is a major bottleneck: input-only hidden states do not reliably identify which relation behavior is requested. With a learned router, explicit FVs lose gold likelihood against oracle routing, and the compact expert code adds another large quality loss. Shared-pool bookkeeping also fails to beat global sparse lists in bytes.

**Hypothesis:** Explicit task context, more support data, a query-conditioned router trained for the task family, or separate expert targets may help. These are new designs and cannot be tuned against the sealed seeds.

## C — Strongest counter-hypothesis

The .477/.516 route accuracy may reflect ambiguous or overlapping input domains (for example forward/reverse relations), not a limit of SAE-feature experts. Even with this router, however, the frozen deployable path misses its gates, and the oracle/learned FV gap isolates meaningful routing cost.

## U — Boundaries

One small LM, one SAE checkpoint and relation-task queries only. This does not establish learned feature-expert routing quality on natural MoE workloads, additional model capacity, or optimized runtime.
