# MA-001 — nonlinear top-1 Mirror experts

Status: SCREENING  
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

Pending development protocol execution. No fresh rows may be written before the development gate decision and fresh freeze.

## Decision

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
BOUNDARY: synthetic 16D routed regression; no natural-language or independent-capacity claim.
