# Wireless Mirror family diagnostic — 2026-11-13

## Decision

Pause MA-986 through MA-989 for structural redesign. Keep their registry statuses **UNTESTED**. Resume the family only when a candidate specifies a Mirror coordinate whose function or measured Pareto frontier is distinct from the strongest native codebook, Type-II, embedding, phase-gradient, or product-code control.

MA-990 is a separate spatial-audio family and proceeds next.

## Evidence

| Candidate | Observed fact | Scoped outcome |
|---|---|---|
| MA-981 | Sector Mirror phase address and native steering code matched exactly. Mirror payload 10,401 B; independent 16-beam banks 18,277 B, so the <=50% gate was missed at 56.9%. | FAIL; synthetic ULA only. |
| MA-982 | Mirror site phase coordinate and native phase-parametric Type-II matched exactly. Model+feedback reduction versus stored per-site Type-II was 2.9%, below the 20% gate. | FAIL; synthetic channel model only. |
| MA-983 | Rank-1 domain Mirror and ordinary embedding had identical payload hash, function and metrics. Independent decoder quality was substantially higher. | FAIL; synthetic linear CSI compressor only. |
| MA-984 | Two-coordinate phase codes retained 99.73% of the continuous rate and reduced commands versus full-element 2-bit control, but native phase-gradient control exactly matched the function and bytes. | FAIL for Mirror-specific attribution; useful ideal phase-code boundary case. |
| MA-985 | Rank-2 AP×RIS factorization and native product control had identical payload hash and WSR; both missed quality and storage gates. | FAIL; per-configuration synthetic optimization only. |

## Fact / interpretation / hypothesis

- **Fact:** MA-981 and MA-982 are consecutive wireless P0 screens whose Mirror functions are exactly reproduced by native steering/Type-II coordinates. MA-983, MA-984 and MA-985 independently repeat the native-alias pattern through ordinary embedding, phase-gradient, and product-code controls.
- **Interpretation:** the current wireless proposals mostly rename established codebook, phase-angle, Type-II, or product-factor coordinates. Compressing state against a dense table is not a Mirror-specific contribution if a native parametric control provides the same function at the same cost.
- **Hypothesis:** the wireless family may become testable again with a genuinely different coordinate that improves a matched native frontier on measured channels, counts site/user feedback and pilot/control overhead, and demonstrates useful quality on unseen domains.

## Boundaries

These results use synthetic channels or idealized phase models. They do not reproduce PA288, PA289, PA290, CsiNet, 3GPP propagation, a deployed network, RF energy, or hardware switching latency. No evidence is pooled across the five experiments because their tasks and metrics differ.
