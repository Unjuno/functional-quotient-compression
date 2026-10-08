# MA-255 — Mirror context superposition for task models

Status: SCREENING (protocol frozen; development pending)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA16, Parameter Superposition.

## H — falsifiable hypothesis

For related linear task functions stored in one shared physical matrix, learned task-specific Mirror views (diagonal and orthogonal) will recover all task functions with lower serialized payload than independent matrices, and beat fixed random PSP contexts and a byte-near low-rank task-code control at comparable active compute. Failure to beat those controls falsifies Mirror-specific advantage here.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact task coordinate `m_t` to the output interface of a shared physical linear map, so task-specific diagonal or orthogonal functional views can be expressed without storing one full matrix per task.

- Physical object: one D x O shared matrix.
- Coordinate: task code decoded as output-channel diagonal gains (Mirror-diagonal) or matrix-free Givens rotations (Mirror-orthogonal).
- Logical multiplicity: T task-specific linear maps.
- Native method: Parameter Superposition (PSP), fixed Rademacher contexts and unbinding.
- Controls: common shared model; task-code-conditioned rank-2 additive residual; independent matrices upper reference.

## Comparisons

`independent`, `shared`, native fixed-context `psp`, `lowrank`, `mirror_diag`, and `mirror_orthogonal`. Same examples and update budget. Fixed-update results are mechanism/learning evidence only.

## Gates

- **PASS (screen only):** a Mirror method gets each task's held-out MSE within 1.10x independent, uses <=50% independent bytes, and beats PSP and lowrank by >=10% mean MSE at payload within 15% or strictly lower bytes.
- **FAIL:** Mirror misses quality/byte gate or PSP/lowrank matches at equal/lower bytes.
- **NOT ESTABLISHED:** unstable execution, serialization/replay failure, or controls fail expected training regime.

## Tuning boundary

Development world 25500, init seed 255000; LR {0.003, 0.01}, 1200 updates. Choose common LR by minimum mean task-heldout MSE across methods. Fresh worlds 25501–25503 with init seeds 255001–255003 are frozen; open only if development is promising.

## Storage and compute

Authority: actual `torch.save` inference state plus canonical JSON metadata. Count all learned tensors and codes. Report examples, updates, active MAC proxy, training wall time and inference throughput separately.

## Boundaries

Synthetic linear family only; no language model, near-convergence fixed-byte frontier, task inference/routing, or general capacity claim. Task ID is supplied at inference.

## D — FAIL (development screen)

The preregistered selector chose LR 0.01 (mean MSE across methods 0.418932 versus 0.418673 at LR 0.01). The best Mirror method was `mirror_orthogonal`: mean task MSE 0.0921874, worst-task MSE 0.114966, 3668 actual payload bytes. Independent upper control: mean MSE 4.52389e-05, 10364 bytes. Mirror payload is 35.4% of independent, but its mean MSE is 2038x worse, failing the quality gate.

At the selected LR, lowrank scored mean MSE 0.13569 at 5616 bytes. Mirror-diagonal scored 0.196301 at 3896 bytes. Orthogonal Mirror improved on these controls in this fixed screen, but remained far from independent-task quality and used more active compute/wall time than lowrank. No Mirror-specific Pareto improvement was established.

### C — strongest counter-hypothesis

The task generator uses dense random output rotations and channel gains, while the learned Givens view is a fixed sequence of adjacent-plane rotations after a shared map; finite updates and one optimization screen may limit recovery. Conversely, rank-2 residual captures substantially more quality than diagonal Mirror at greater but still sub-independent storage.

### U — not established

Fresh-world replication, near-convergence capacity, fixed-byte frontier, natural-language quality, and whether a better aligned transform family can close the gap remain untested. Fresh seeds were not opened because the development gate failed.

## Fact / interpretation / hypothesis

- **Fact:** The development screen has 12 rows (six methods x two rates); at selected LR 0.003, best Mirror MSE is 0.0921874 and independent MSE is 4.52389e-05. Actual serialized payloads were measured from `torch.save` state plus metadata.
- **Interpretation:** These Views reduced payload but did not recover task functions sufficiently; no Mirror-specific advantage over the simpler low-rank control was established.
- **Hypothesis:** The restricted post-map coordinate family or optimization budget limited recovery; a separate preregistered experiment would be needed to test that.

### PSP implementation audit

The initial dev pass exposed a PSP control implementation error: it used the shared model tensor as every task's bound item, while also serializing the fixed context tensor. Before any fresh data access, the protocol was amended: the PSP baseline now trains one physical superposed tensor directly and reconstructs seeded Rademacher contexts from paid seed metadata rather than serializing the context table. The development screen was rerun from scratch. This corrected control remained poor (mean MSE about 1.77), but its result remains an approximate implementation screen and is not presented as a reproduction of the full PA16 paper.
