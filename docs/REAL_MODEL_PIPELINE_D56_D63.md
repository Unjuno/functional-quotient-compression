# D56–D63 as the Canonical Real-Model Pipeline

The D56–D63 historical line is best understood as one coherent pipeline for turning a real Transformer checkpoint into an auditable FQC compression experiment.

1. **D56 — Real-pilot contract.** Freeze the baseline storage boundary, integer 64× budget, decoder protocol, numeric contract, artifact hashes, and PAID/EXTERNAL_FIXED/DERIVED classes.
2. **D57 — Architecture-correct extraction.** Inventory unique storage and extract only mathematically valid functional primitives. RoPE, GQA/MQA, QK normalization, tying, and MLP symmetries are explicit.
3. **D58 — Structural diagnostics.** Measure shared span, support concentration, commutators/AJD only where valid, RoPE atom reuse, GQA consumer reuse, and symmetry-normalized MLP similarity. Diagnostics are not codec claims.
4. **D59 — Diagnostic-to-codec compilation.** Convert proposed shared structure into a paid-atom decoder DAG and charge the union dependency closure exactly once.
5. **D60 — Rate–distortion frontier.** Optimize codec candidates using actual reconstruction distortion and full candidate rate rather than diagnostic residual alone.
6. **D61 — Functional sensitivity.** Weight local reconstruction perturbations by downstream functional effect instead of Frobenius error alone.
7. **D62 — Sensitivity-aware allocation.** Solve the integral hard-budget allocation problem; do not assume density-greedy is exact.
8. **D63 — Shared-atom allocation.** Jointly open shared roots/dictionaries/bases and assign blocks. Fixed shared cost is amortized over the coalition and charged by union closure.

## Canonical interpretation

D56 and D59 should be read together. D56 defines the accounting contract; D59 supplies the stronger graph compiler needed to enforce it. The canonical repository therefore uses one decoder-DAG implementation with full dependency-cycle validation and union accounting.

D62 is separable only when no conditional shared paid atoms exist. D63 is the more general FQC case and should become the allocation core once candidate shared roots are introduced.

## Root-candidate measurements for the real pilot

For every proposed root/dictionary/basis candidate, record at minimum:

- fixed paid closure bits;
- the set of blocks that can consume it;
- per-block private-rate saving relative to the private alternative;
- per-block functional error change;
- smallest pure-rate break-even coalition;
- exact hard-budget optimum after joint coalition selection.

A root that is unattractive for every block in isolation may still be globally useful after its fixed cost is amortized. Conversely, ignoring the fixed shared cost can create false hard-budget passes.

## Evidence boundary

D56–D63 contain contracts, exact arithmetic, synthetic diagnostics, and small allocation examples. They do **not** establish that a real Transformer has enough shared structure for 64× compression. The real pilot must populate these interfaces from an actual checkpoint and serialize the resulting codec.
