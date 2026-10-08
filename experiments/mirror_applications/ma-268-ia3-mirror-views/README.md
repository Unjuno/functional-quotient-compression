# MA-268 — IA3 Mirror activation views

Status: **PROMISING (trained nonlinear aligned task screen; a separate post-fit linear screen missed its strict byte gate)**
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA19 IA3/T-Few.

## H — falsifiable hypothesis

A small task-specific hidden rotation coordinate over a shared nonlinear feature network will recover role-specific functions better per actual byte than native IA3 diagonal activation scaling. If IA3 matches at equal/lower bytes, richer Mirror coordinates have no specific benefit here.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact task coordinate `m_t` at the shared hidden activation interface, applying a fixed-width Givens rotation before the shared output map to express logical task views without private task networks.

- Shared object: 16x16 hidden network and shared 16x4 readout.
- IA3 control: learned task-specific hidden-channel gains.
- Mirror: task-specific four-angle block Givens rotation.
- Independent full task networks: upper reference.

## Protocol

16D input, width 16, four classes/tasks, tanh shared hidden feature map. Teacher functions use task-specific hidden-space Givens rotations plus independent output heads. Methods: independent task networks, hard-shared network, IA3 hidden scales plus shared readout, Mirror Givens plus shared readout. Dev worlds 26800/26801; LR {.003,.01}, 1,000 updates, batch 128. Fresh 26802–26804 sealed unless dev passes.

**PASS (screen only):** Mirror MSE <=1.10x independent, bytes <=65% independent, and beats IA3 by >=10% MSE at byte-near cost (within 15%) or lower bytes. **FAIL:** gate missed or IA3 matches at equal/lower bytes.

Report per-task quality, actual serialized bytes, active compute, wall time and throughput.

## Boundaries

Synthetic regression only, fixed teacher family, no language or capacity claim.

## D — PROMISING (bounded synthetic result)

Development selected LR 0.01. On all 3 fresh worlds, Mirror mean MSE was 0.0049–0.0060, versus IA3 0.0143–0.0246 and independent 0.0057–0.0069. Mirror used 4,167 serialized bytes, 2.5% less than IA3 (4,273 B) and 47.5% less than independent (7,933 B). It passed the registered quality/byte gate in this task family.

Compute is a trade-off: Mirror's active MAC proxy is 147,456,000 per training screen, 1.5x IA3 and 45% of independent. CPU inference throughput was ~2.05–2.46M examples/s for Mirror, ~7.11–7.49M for IA3, and ~1.15–1.30M for independent. The unoptimized eager Givens implementation is slower than IA3, so no runtime benefit over efficient channel scaling is claimed.

### C — strongest counter-hypothesis

The teacher was deliberately generated from task-specific Givens hidden rotations, so this is an aligned mechanism feasibility result. It does not show that rotations beat IA3 on natural tasks or arbitrary task families. The 106-byte storage difference from IA3 is small.

### U — not established

Natural language, learned routing, near-convergence fixed-byte capacity, GPU fused-kernel runtime, robustness to task distributions not aligned to Givens, and transfer to unseen task codes remain untested.

### Fact / interpretation / hypothesis

- **Fact:** Mirror passed the declared quality/byte gate in all 3 fresh worlds; every method used the frozen LR 0.01 and 1,000 updates. Actual serialized payloads and MAC proxies were measured.
- **Interpretation:** A compact activation View can recover this deliberately rotation-structured task family using about half the independent-model bytes, with slower execution than IA3.
- **Hypothesis:** Mirror's benefit depends on task variation being rotational rather than diagonal; a future cross-over task could map the boundary.

## Protocol variant reconciliation

The distinct `protocol_variants/postfit_linear_scale_orbit/` directory preserves another frozen MA-268 study. It used a post-fit frozen 12x6 linear block and four fresh seeds. Mirror MSE was 2.70e-24 versus IA3 2.12e-24, but the actual payload saving was 15.7% (1,550B vs 1,838B), below that variant's 20% byte gate; on independent random gains Mirror MSE was 0.210 while IA3 remained near 2.78e-24. That variant is a bounded near-miss, not a PASS.

The root result is PROMISING because the separate nonlinear trained screen passed its own frozen quality/byte gate in 3/3 fresh worlds. Both task families are synthetic and deliberately aligned to different extents. Across the primary screen, Mirror improves quality-per-byte over IA3 but uses 1.5x its active-MAC proxy and only about 0.29–0.34x its eager CPU throughput, so no runtime or broad activation claim follows.

Source branches: trained nonlinear screen `research/ma-268-ia3-mirror-views-20261008` (result commit `1000a42fe8cff1cb171607ed71b8a0b33ee4e17b`); post-fit linear screen `research/ma-268-ia3-mirror-activation-20261008` (verified commit `ee133120629a67f41387ff658e12acf504ccbbca`).
