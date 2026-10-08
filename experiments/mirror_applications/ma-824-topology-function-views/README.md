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
