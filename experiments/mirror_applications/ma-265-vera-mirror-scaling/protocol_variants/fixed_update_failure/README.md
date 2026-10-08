# MA-265 — VeRA Mirror scaling code bank

Status: SCREENING (protocol frozen)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA18 VeRA.

## H — falsifiable hypothesis

With the same frozen random low-rank basis and comparable task-code bytes, a structured Givens Mirror coordinate over VeRA's scaling interface improves held-out regression quality or useful adapter multiplicity over native VeRA scaling vectors. If VeRA matches or exceeds the Mirror frontier, Mirror-specific value is falsified for this task family.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact task coordinate `m_t` to VeRA's shared frozen low-rank basis coefficients, allowing each logical adapter to rotate coefficient channels without storing independent low-rank bases.

- Physical object: frozen shared random A/B basis, rank 4.
- Native VeRA coordinate: learned task-specific left/right scale vectors.
- Mirror coordinate: task-specific Givens mixing of rank channels, with same shared basis.
- Controls: independent rank-4 adapters, zero adaptation/shared base, native VeRA scaling.

## Protocol and gates

Sixteen-dimensional inputs, eight outputs, four tasks. Teacher is shared base plus task-specific low-rank rank-2 delta in the shared rank-4 basis. Development worlds 26500/26501, LR {0.003,0.01}, 1,000 AdamW updates, batch 128. Fresh 26502–26504 remain sealed unless a dev gate passes.

**PASS (screen only):** Mirror mean MSE <=1.10x independent, payload <=60% independent, and >=10% lower MSE than VeRA at byte-near storage (within 15%) or lower bytes. **FAIL:** gate missed or VeRA matches at equal/lower bytes.

Count frozen random basis in payload; only deterministic seed-generation metadata may be free if exact reconstruction is verified. Report actual payload, active MAC proxy and runtime.

## Boundaries

Synthetic linear adapter task family; no natural-language or near-convergence fixed-byte capacity claim.

## D — FAIL (development screen)

The preregistered selector chose LR 0.01 by mean MSE across methods/worlds. Across the two development worlds, native VeRA outperformed Mirror: mean MSE about 0.117 versus 0.232. Independent adapters were near-exact (mean about 6e-5). Mirror used 3,417 actual bytes versus VeRA's 3,700, a small storage reduction with a substantial quality loss. Mirror failed the quality gate and did not establish a useful Pareto improvement. Fresh worlds 26502–26504 were not opened.

### C — strongest counter-hypothesis

The teacher uses task-specific diagonal rank coefficients in the same frozen basis; VeRA scaling is directly matched to that structure, while the tested Givens mixing rotates paired rank channels and cannot independently recover arbitrary per-channel scaling. This is a deliberately strong VeRA-native task family and the result is scoped accordingly.

### U — not established

Fresh replication, near-convergence fixed-byte frontier, nonlinear adapters, and Mirror views on task families not aligned with VeRA's scale vectors remain untested.

### Fact / interpretation / hypothesis

- **Fact:** Two dev worlds x two learning rates x four methods were run with common data/basis per world and 1,000 updates. Payloads include the shared frozen basis.
- **Interpretation:** On this basis-aligned family, ordinary VeRA scales were more effective than the tested Mirror rotation; the ~8% byte saving did not compensate for worse regression quality.
- **Hypothesis:** Mirror may add value when task variation is rotational/compositional rather than diagonal scaling, but that requires a separate preregistered candidate.
