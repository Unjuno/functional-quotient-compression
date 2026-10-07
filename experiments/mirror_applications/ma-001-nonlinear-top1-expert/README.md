# MA-001 — nonlinear top-1 Mirror experts

Status: **PROMISING** (fresh aligned sharing gate passed 3/3; Mirror-specific storage gate missed vs ordinary tying)
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME  
Branch: `research/ma-001-mirror-top1-expert-20261007`  
Base commit: `fa635d0` (verified MA-111 parent)

## H — hypothesis

For a four-role top-1 MoE whose nonlinear expert functions share one physical FFN under role-specific orthogonal input/output coordinates, one shared FFN plus four charged Mirror addresses will recover held-out routed quality with fewer actual serialized bytes than independently stored FFNs. It must beat hard tying and byte/compute-aware FiLM or low-rank residual controls to establish a Mirror-specific mechanism. Independently generated role FFNs should require private weights.

## Scope and prior-art delta

- **MA-003 already tested** this top-1 claim for linear maps. This experiment extends it to a two-linear-layer GELU FFN and is not counted as a replication of MA-003.
- **MA-241 already tested** a nonlinear expert pool tied across depth with layer-specific views and soft routing. Here there is one MoE layer, hard top-1 dispatch, and expert-specific views within the layer.
- **PA01:** ordinary tying must be a control. **PA02:** no cross-layer path constraint exists in this one-layer task; routers remain learned and role-specific only by their expert scores. **PA03:** rank-factorized routing must be included alongside dense routing.
- Exact Mirror delta: separate Givens address per logical expert conjugates one nonlinear FFN. Simpler controls include a shared hard-tied expert, per-expert hidden FiLM, and per-expert rank-2 output residual.

## Physical-to-logical claim

- Physical object: one `16 -> 32 -> 16` GELU FFN.
- Address: four learned four-angle Givens codes, applied to the 16D input and inverted on the 16D output.
- Logical multiplicity: four router-addressable nonlinear expert behaviors with top-1 execution.
- Failure risks: views may not approximate useful expert diversity; FiLM/residual can be better or cheaper; independent experts need more private state; eager coordinate transforms may lose wall-clock.

## T — protocol

See frozen `PROTOCOL.json`. One development world chooses a common learning rate over every method and teacher mode. Three independent fresh worlds are accessed only if the development screen meets the preregistered relative-quality and byte gates. Fresh configuration will be frozen after development and before fresh data generation.

This is a fixed-update synthetic mechanism screen, not a language-model, capacity, or convergence claim. The aligned teacher is deliberately favorable to an orthogonal-view model; the independent teacher probes the private-parameter boundary.

## Storage and compute

Authoritative storage is the actual serialized `torch.save` inference payload plus deterministic config metadata. It charges the router, FFN, all per-role view/FiLM/residual state, and reconstruction metadata. Compute reports the top-1 active expert MAC proxy, router operations, coordinate operations, updates, examples, wall time, and measured CPU inference throughput.

## Results

Development selected LR 0.01 by the preregistered common mean-MSE rule. At this setting, aligned development Mirror MSE was 0.000764 vs 0.001404 for dense untied MoE, and payload was 7,697B vs 23,277B; the fresh-access gate passed. Fresh worlds 10001–10003 then ran at the frozen setting. All 36 fresh result rows were replayed; deterministic outputs, metrics, compute proxies and serialized byte counts matched exactly.

### Fact

- Aligned fresh Mirror routed MSE was 0.000484 / 0.000672 / 0.000554, or 0.419x / 0.572x / 0.491x dense untied MoE. Router accuracy was within 0.40 percentage points of dense MoE in every world. The registered useful-sharing quality/storage gate passed 3/3.
- Mirror payload was 7,697B, 0.331x dense untied MoE (23,277B), and 0.329x rank-2-router MoE (23,466B). It was 252B (3.4%) larger than ordinary tying (7,445B), while substantially improving held-out MSE over tying.
- Mirror MSE was 0.102–0.112x hard tying, 0.169–0.197x FiLM, and 0.152–0.177x rank-2 residual across the three worlds. However, Mirror's payload exceeded hard tying, so it failed the preregistered Mirror-specific byte condition. Its active MAC proxy was 1,088, plus 48 coordinate FLOPs per example; rank-2 residual used 1,152 MACs.
- On the independent-role teacher, Mirror MSE was 0.0184–0.0215 vs 0.00292–0.00373 full MoE; FiLM reached 0.00761–0.00872. This marks a boundary: small Givens views did not recover unrelated nonlinear functions.
- Runtime cost was material in this CPU eager implementation. Median aligned training wall time was 9.85s for Mirror vs 1.81s hard tying; median inference throughput was 0.538M vs 2.769M examples/s (0.194x). Throughput is a single-host implementation measurement, not a hardware-general result.
- Three tests passed. The fresh replay matched all 36 deterministic result rows exactly; wall-clock and throughput fields are expected to vary and were excluded from deterministic replay comparison.

### T — execution

Six methods (dense untied, PA03-style rank-2 router, ordinary hard tying, tied+FiLM, tied+rank-2 residual, tied+Mirror) trained for 1,200 updates × 64 examples on matched data in aligned and independent teacher modes. Development world 10000 selected LR 0.01. Fresh worlds 10001–10003 used frozen seeds and settings. Actual bytes are serialized CPU state dict plus config; router, views, controls and metadata are included. The frozen protocol and sources were hash-checked before fresh access.

### D — decision: PROMISING

The registered 3/3 useful-sharing gate passed on a deliberately view-aligned nonlinear expert family, with 66.9% fewer payload bytes than dense MoE. Mirror strongly improved quality over FiLM and residual controls. It does not establish a Pareto win over the smallest hard-tied payload: Mirror costs 252B more, and CPU throughput is roughly 5.2x lower. Thus the experiment supports a narrow aligned functional differentiation result, not a Mirror-specific storage advantage over every simpler control.

### C — strongest counter-hypothesis

The teacher is constructed from the same Givens-conjugacy family used by Mirror and the training budget is fixed rather than near-converged. This favors the candidate. A generic shared basis or a learned low-rank control with equalized bytes/compute could narrow the quality gap; that comparison is not established here. The coordinate implementation also adds expensive eager tensor operations.

### U — unconfirmed

Natural-language MoE quality, near-convergence/fixed-byte capacity, GPU kernels, top-k greater than one, load balance, more general view families, equal-payload generic basis controls, and multi-layer router/path behavior remain untested. Independent functions demonstrate one private-parameter boundary but do not map the full boundary.

## Evidence separation

- **Fact:** values and gate outcomes above are in `RESULTS_CORE.csv`; payload sizes are measured serialized bytes.
- **Interpretation:** a small structured view can usefully differentiate one physical nonlinear expert for a deliberately aligned family at lower bytes than untied MoE.
- **Hypothesis:** related natural expert functions may expose similar low-description coordinates; this experiment does not establish that transfer.
