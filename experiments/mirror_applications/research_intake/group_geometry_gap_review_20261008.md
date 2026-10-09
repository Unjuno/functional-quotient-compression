# Research intake — group geometry gaps (2026-10-08)

Status: RESEARCH ONLY. No MA IDs reserved, no worker queue modification, no claimed experiment result.

## Registry audit

Current branch registry has 1115 entries through MA-1115. Existing group action MA-331..338, Cayley MA-707, Householder MA-701..702, gauge MA-1115 and function-space review must be used as controls; avoid re-registering their general hypotheses.

## Four candidate gaps for future deduplication

1. **Stabilizer-aware code capacity**: measure effective distinct functions per stored Mirror bit after quotienting transformations that leave a calibrated function invariant. Controls: MA-332..335, MA-1115, native shared basis. Distinct parameter orbits are not automatically distinct functions. H: a stabilizer-quotiented code reduces inference bytes at matched quality vs unconstrained code. FAIL if bytes or held-out task quality fail to improve. The cost of estimating the stabilizer is offline training cost, not free.
2. **Noncommuting Mirror composition order**: compare ordered compositions AB vs BA on held-out task pairs at identical code bytes and FLOPs, against additive task vectors, MA-257 and native matrix products. H: learned ordered task composition improves unseen ordered-pair loss without private task-pair adapters. FAIL if native products or additive baselines match. Record any extra multiplication latency.
3. **Gauge-robust alignment without oracle task deltas**: learn small codes from target training examples, align source-task function responses on a disjoint calibration set, then compare against Procrustes-aligned factors, SVD, BOLT, MA-1096 and MA-1115. H: learned alignment remains invariant under invertible LoRA rank-coordinate reparameterizations while beating a byte-near native alignment. FAIL on gauge instability, label leakage, or native-control parity.
4. **Equivariant cache transport under role transforms**: evaluate when a shared canonical attention state can be exactly transformed across roles without target prefill; include noncommuting position/role transformations and numerical precision effects. Controls: MA-691..700, MA-1112, native re-prefill, physical aliasing. Distinguish exact algebraic transport from approximate cache translation; do not count a copy as shared VRAM.

## Experimental discipline

For every gap: freeze data splits and controls before audit, measure actual serialized inference bytes (basis, router, code, metadata), active FLOPs, P50/P95 wall latency, quality and OOD error; use >=3 fresh seeds for an initial mechanism screen, increase replication if uncertain. Separate fixed-update learning efficiency from converged capacity. Run registry integrity check only if registry or status/claim files are changed. No current candidate or priority was changed.

## Next decision

Deduplicate these four gaps against full descriptions, not keyword counts, before assigning any new MA IDs. Prioritize a single cheap falsification screen; no novelty claim is established by this note.
