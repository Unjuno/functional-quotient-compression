# MA-824 — factorized topology × node-function views

Status: SCREENING — frozen compositional serialization screen
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Selection: Draw22, uniform over eligible P0/UNTESTED candidates; replay in `source/random_draw.json`.

## H — Hypothesis

A shared node-function bank and a small product of topology and function addresses can execute held-out graph/function combinations with fewer actual bytes than duplicated complete programs. A PA128 Neural Interpreter-style signature is the strongest native control; parity with that control means the result is generic program factorization, not Mirror-specific.

## Mirror insertion

> **Mirror insertion:** this experiment adds separate topology and node-function address factors to a shared two-node graph substrate and operator bank, so logical neural programs can be composed without duplicating their node weights.

- Physical object: two fixed 1→4→1 node MLPs and a two-node graph substrate.
- Mirror coordinates: topology address × node-function address.
- Logical combinations: two graph topologies crossed with two node functions.
- Prior art: PA220 GrapNet and PA128 Neural Interpreters.
- Controls: duplicated independent programs, explicit graph plus function signatures, hard shared indices.

## T — Frozen task

The node MLP weights are deterministic and frozen. Each program composes either `f(f(x0))` or `f(x0)+f(x1)` using one of two functions. All four compositions are evaluated; one topology/function pair is withheld from the program manifest and constructed from the two observed factors. There is no optimizer training: this screen measures execution, exactness and payload accounting for a small known library. Conditions and gates are in `PROTOCOL.json`.

## Boundaries

The result can establish exact composition and serialization efficiency for four synthetic programs. It cannot establish learned neural interpreter quality, arbitrary logical capacity, or Mirror-specific value if the native interpreter stores the same factorization at equal cost.

## Result and worker report

**Fact.** With independently serialized node tensors, all four programs including the held-out `parallel × sine_mlp` pair had exactly 0 fresh MSE in all three worlds. Mirror payload was 3,780 B, versus 10,778 B for duplicated independent programs (64.9% fewer bytes), 3,956 B for the PA128-style native interpreter (4.45% fewer), and 3,840 B for hard shared indices. The Mirror-specific 10% advantage over the native interpreter failed in all three fresh worlds. The initial aliased-storage attempt is separately preserved and excluded from this result. Twenty amended payloads passed hash and output replay with max error 0.0; no optimizer updates occurred and no audit set was opened.

**Interpretation.** Separating topology from node-function identity composes the tested factors and saves bytes against physical duplication. The PA128-style shared interpreter achieves the same exact functions with nearly the same storage, so the observed compression is generic program sharing rather than a Mirror-specific gain. The four tested programs are not a capacity count.

**Hypothesis.** Larger graphs or a broader function library may make separate topology/function codes useful if a native program signature must store repeated routing detail. This small exact executor does not establish that advantage.

**H** Factorized topology × node-function codes would compose held-out combinations and compress duplicated program state, with a measurable storage advantage over native interpreter signatures. **T** Two dev and three amended fresh seeds, two topologies, two fixed node MLPs, four controls, actual serialized payloads and held-out output replay; no learning. **D: FAIL** for Mirror-specific value: independent duplication was compressed, but the native interpreter advantage missed its frozen 10% threshold. **C** Neural Interpreter/GrapNet-style generic code factoring explains the result. **U** Learned node libraries, larger graph families, natural programs and capacity remain untested.
