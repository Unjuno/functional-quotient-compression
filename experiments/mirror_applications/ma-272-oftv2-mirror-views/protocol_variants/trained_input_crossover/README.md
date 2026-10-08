# MA-272 — input-centric OFTv2 Mirror views

Status: SCREENING (protocol frozen)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA22 OFTv2.

## H — falsifiable hypothesis

Input-side compact Givens Mirror coordinates can express four logical task transforms over one shared linear map with lower actual payload and active compute than dense input-centric OFTv2, while retaining independent-task quality. If native OFTv2 is as compact/fast or Mirror loses quality, the hypothesis fails.

## Mirror insertion

> **Mirror insertion:** this experiment adds a task coordinate `m_t` at the input activation interface, applying a matrix-free orthogonal View before one shared physical linear map, avoiding one dense input transform per task.

- Physical object: shared 16x16 linear map.
- Native control: OFTv2-style per-task input-side orthogonal transform, applied as `x @ Q_t` before shared W; no transformed weight materialization.
- Mirror: eight learned input Givens angles per task.
- Controls: independent task maps, hard shared map, native dense OFTv2 transforms.

## Protocol

Four tasks, 16D input/output. Teacher maps use shared W after task-specific input-side Givens transform. Dev worlds 27200/27201; LR (0.003, 0.01), 1,000 updates, batch 128. Fresh 27202–27204 sealed unless dev passes.

**PASS (screen only):** Mirror MSE <=1.10x independent, payload <=60% independent, and lower actual bytes than native OFTv2 with <=1.10x its MSE; report active MAC and CPU throughput separately. **FAIL:** misses quality/byte gate or OFTv2 matches at equal/lower bytes and runtime.

## Boundaries

Synthetic linear task family, fixed update budget, no general OFTv2 or language/capacity claim.

## D — PROMISING, with eager runtime regression

Development selected LR 0.01. Across all three fresh worlds, Mirror held-out MSE was 6.0e-10–8.1e-10; independent maps were also near exact, while native dense input-side OFTv2 ranged 0.013–0.023. Mirror payload was 3,272 B, 54.2% of independent (6,033 B) and 45.6% of native OFTv2 (7,175 B), passing the registered quality/storage gate.

Compute is not uniformly improved: Mirror's MAC proxy was 131.1M versus 196.6M independent and 2.16B for this dense Cayley OFTv2 implementation. Eager CPU throughput was 0.77–0.83M examples/s for Mirror, below OFTv2 at 1.16–1.24M and independent at 1.78–1.87M. Thus storage and quality improved in this aligned mechanism screen, while actual eager latency regressed versus native input-side OFT. The OFTv2 arithmetic proxy includes dense Cayley solves and is not a fused-kernel benchmark.

### C — strongest counter-hypothesis

The teacher is generated from the same compact input Givens family as Mirror, so this is a strongly aligned feasibility screen. Independent maps also fit well and are only ~1.84x Mirror bytes, not 4x, because the task maps are serialized compactly in this fixture.

### U — not established

Arbitrary orthogonal transforms, non-linear layers, natural-language quality, fused/GPU runtime, near-convergence fixed-byte frontier, and unseen task coordinates remain untested.

### Fact / interpretation / hypothesis

- **Fact:** At frozen LR=0.01, Mirror passed quality/byte gates in 3/3 fresh worlds with serialized payload 3,272 B; the code stores shared W plus 32 angles.
- **Interpretation:** Input-side compact Views can recover this rotation-aligned family more compactly than dense OFTv2 transforms, but the current eager implementation is slower.
- **Hypothesis:** A fused input-rotation kernel may preserve storage/quality while reducing latency; alignment dependence remains to be mapped.
