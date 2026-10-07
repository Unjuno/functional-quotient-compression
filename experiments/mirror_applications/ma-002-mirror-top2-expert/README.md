# MA-002 — nonlinear top-2 Mirror experts

Status: **PROMISING** (aligned top-2 quality/storage gate passed 3/3; Mirror-specific compute/storage gate missed vs hard tying)
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-002-mirror-top2-expert-20261007`  
Base commit: `4236bb6` (verified MA-001 parent)

## H — hypothesis

For a learned top-2 MoE whose nonlinear expert functions are related by per-role Givens coordinates, one shared FFN with charged role-specific views can recover the teacher's probability-weighted two-expert output at lower serialized bytes than untied top-2 MoE. The view model must beat hard tying and byte-near FiLM/rank-2 residual controls. Independent role FFNs test when private weights become necessary.

## Prior-art delta

- **MA-003** tested hard top-1 routing over linear Givens experts; **MA-001** tested nonlinear hard top-1 experts. This experiment changes the functional composition: two router-selected roles contribute with their normalized router weights for each example.
- **MA-005** evaluated all four linear views under a deterministic signed Walsh mixture. Here routing is learned, sparse top-2, input-conditioned, and nonlinear; it is not an all-expert signed mixture.
- **PA01** requires ordinary tying; **PA02/PA03** require a factorized router control. This single-layer experiment does not test path constraints.

## Physical-to-logical claim

- Physical object: one 16→32→16 GELU FFN.
- Address: four per-role four-angle Givens codes applied before and inverted after the shared FFN.
- Logical multiplicity: four top-2 router-addressable roles; two weighted role outputs are active per input.
- Controls: dense untied top-2 FFNs, rank-2 factorized-router untied FFNs, hard tying, tied+FiLM, tied+rank-2 output residual, and tied+Mirror.

## T — frozen mechanism screen

The protocol is in `PROTOCOL.json`. Development world 20000 selects one LR for all methods and teacher modes; fresh worlds 20001–20003 run only if the preregistered development gate passes. The teacher router is a frozen linear score model whose full softmax is distilled by the learned student router. Top-2 roles and normalized selected probabilities define the teacher/student mixture. Aligned teacher experts are Givens conjugates of one frozen nonlinear FFN; independent mode uses four separately initialized FFNs.

## Storage and compute

Storage is actual serialized CPU state dict plus deterministic JSON metadata, charging router, expert/view/control state and reconstruction metadata. Report two-active-expert MAC proxy, router/coordinate operations, updates/examples, wall time, and CPU inference throughput.

## Results

Development selected LR 0.01 by the preregistered mean-MSE rule; the aligned Mirror/dense MSE ratio was 0.247 and the payload ratio was 0.331, so the fresh gate passed. Fresh worlds 20001–20003 ran under the frozen protocol. All 36 deterministic fresh rows replayed exactly; actual serialized bytes and model outputs round-tripped exactly.

### Fact

- Aligned top-2 Mirror routed MSE was 0.0000608 / 0.0000996 / 0.0001537, or 0.223x / 0.256x / 0.403x dense untied top-2 MoE. Top-2 set accuracy was 1.000 / 0.9995 / 1.000 vs 0.9995 / 0.9985 / 1.000 for dense MoE. The preregistered useful-sharing quality/storage gate passed 3/3.
- Mirror serialized payload was 7,697B vs 23,277B dense MoE (0.331x) and 23,466B rank-2-router MoE (0.328x). It was 252B larger than hard tying (7,445B) and 1,148B smaller than FiLM or rank-2 residual (8,845B each).
- Mirror MSE was 0.037–0.144x hard tying, 0.055–0.191x FiLM, and 0.048–0.226x rank-2 residual across the worlds. Yet the registered Mirror-specific Pareto gate failed vs hard tying: Mirror is larger and has roughly 2.03x its active MAC-plus-coordinate proxy (2,208 vs 1,088 per example). Against untied top-2 the proxy was 2,208 vs 2,112, with the difference coming from 96 coordinate FLOPs.
- Independent-role MSE for Mirror was 0.00632–0.00670 vs 0.000270–0.000504 dense MoE; FiLM was 0.00318–0.00354. This task family needs private or richer role-specific parameters to reproduce unrelated functions.
- Eager CPU runtime regressed. Median aligned training wall time was 9.40s Mirror vs 1.17s hard tying; median inference throughput was 0.407M vs 4.001M examples/s (0.102x). This is a single-host implementation result.
- Three tests passed. All 36 fresh deterministic fields replayed exactly; timing fields were excluded. Serialization/load output equality was checked on each run.

### T — execution

Six methods (dense untied, PA03-style rank-2 router, hard tying, tied+FiLM, tied+rank-2 residual, tied+Mirror) trained for 1,200 updates × 64 examples on matched data in aligned and independent modes. Development world 20000 chose LR 0.01. Fresh worlds 20001–20003 used independent frozen teacher/data/init seeds. The teacher has a frozen linear routing score model; the learned student router receives soft full-router distillation while only its top two weighted experts execute. Serialized bytes include router, experts/views/controls and metadata.

### D — decision: PROMISING

The top-2 useful-sharing gate passed all three aligned fresh worlds, with about 67% fewer payload bytes than untied MoE and substantially lower MSE. The experiment does not show a Mirror-specific Pareto improvement over hard tying: tying uses fewer bytes and about half the active proxy, while Mirror trades that state and runtime for a large quality gain on this aligned teacher. The independent teacher marks a private-parameter boundary. This is a fixed-update synthetic mechanism result, not a capacity or language-model claim.

### C — strongest counter-hypothesis

The aligned teacher is constructed from Givens-conjugated versions of one FFN, which directly matches the Mirror family. A generic generated/shared-basis control at equal bytes and compute could close the quality gap. The coordinate path is eager and expensive; optimized fused kernels could materially change runtime.

### U — unconfirmed

Natural-language quality, near-convergence fixed-byte capacity, generic shared-basis controls, load balance at scale, more expert counts, optimized GPU kernels, and the full private-parameter boundary remain untested.

## Provenance correction

`PROTOCOL.json` was frozen before development but its `base_commit` field contains a transcription error. Git ancestry shows the protocol commit's actual parent was `4236bb640e1f52fd2abd80fb966c0588eb2ff950`; the frozen protocol commit and hash are preserved. This metadata correction is recorded in `PROTOCOL_AMENDMENT.json`. No training, split, gate, or result was changed.

## Fact / interpretation / hypothesis

- **Fact:** measured metrics and gates are in `RESULTS_CORE.csv`; bytes are actual serialized payloads.
- **Interpretation:** sparse top-2 composition can recover multiple aligned nonlinear expert roles from one physical FFN at much lower bytes than untied MoE, at a substantial eager-runtime cost.
- **Hypothesis:** structured views may be useful when natural experts are related by low-description coordinate changes; this screen does not establish such structure in trained language models.
