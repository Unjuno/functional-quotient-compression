# MA-379 — Mirror-compressed AdapterFusion source bank

Status: FAIL; frozen development gate did not pass.
Evidence lane: QUALITY / STORAGE / COMPUTE / ADAPTER FUSION
Base commit: `ad51e99`
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## H — Hypothesis

A shared adapter matrix plus one Givens angle per source task will retain target fusion quality with fewer bytes than a full AdapterFusion source bank and outperform an equal-byte direct scalar source code.

## Mirror insertion

> **Mirror insertion:** this experiment adds one angle per source-task adapter. The angle conjugates one shared 16×16 adapter basis, creating multiple source functions without four physical matrices.

- Native method: AdapterFusion over four independently trained frozen source adapters.
- Exact insertion: conjugate the shared adapter matrix by a Givens rotation before target fusion.
- Persistent `m`: four learned source angles.
- Sweep: fixed four-source coordinate, no tuning sweep.
- Cheapest ordinary alternative: one direct scalar gain per source adapter.

## Physical-to-logical claim

- Shared object: one 16×16 adapter matrix; target fusion remains task-specific and charged.
- View coordinate: one angle per source adapter.
- Logical multiplicity: four source adapters composed into four target tasks.
- Storage premise: replace four source matrices with one basis plus four codes.
- Failure mode: source tasks may be unrelated matrices outside one conjugacy orbit; scalar control may match the code effect.

## Prior-art delta

Read the registry row and referenced PA items first.

- Closest prior art: PA54 AdapterFusion.
- Prior art establishes: frozen independent source adapters can be fused through learned target-specific attention.
- Exact Mirror delta: compress source adapter storage through structured shared-basis Views without losing target-task quality.
- Cheapest simpler control: scalar source gains on one shared adapter basis.

## T — comparisons and protocol

Compare four independent source adapters with learned target fusion (mandatory AdapterFusion reference), hard shared source basis, scalar-shared source codes, and Givens Mirror source codes. Fit source adapters and fusion jointly against the same source/target matrix task bank; test on fresh examples from the same fixed task world per seed.

## Gates

### PASS
Both development seeds must meet the full quality/byte gate in `PROTOCOL.json` before fresh seeds open.

### FAIL
Mirror target MSE exceeds the registered independent/scalar margin in both seeds or misses actual byte gates.

### NOT ESTABLISHED
Independent AdapterFusion does not learn the target fusion task; fresh stays sealed.

## Tuning boundary

Development: seeds 37901/37902, protocol frozen.

Fresh: seeds 37911/37912/37913; never used for tuning.

## Storage contract

Pay source basis/adapter weights, source codes, target fusion coefficients, task IDs, metadata, and archive overhead. Deterministic ZIP/NPY FP16 inference payload bytes are authoritative.

## Compute contract

Record 1,600 updates, examples, active MACs including Mirror matrix reconstruction, training wall time, and target inference throughput.

## Results

### D — decision

**Fact:** In both development worlds, independent AdapterFusion source maps reached mean target test MSE below 1.1e-8. Mirror target test MSE was 0.00956/0.01032; scalar was 0.00964/0.01023; hard tying was 0.00951/0.01011. Thus Mirror tracked the simple scalar control within the 5% quality margin but was roughly six orders of magnitude worse than independent AdapterFusion. Mirror payload was 2,185B, equal to scalar and 0.625x the independent bank (3,497B), missing the registered 0.60x byte gate; hard tying used only 1,941B. View reconstruction raised the MAC proxy from 1,088 to 50,240 per target example (46.2x). All 8 payloads replayed exactly; 4 tests passed. Fresh remains sealed.

**Interpretation:** The one-matrix conjugacy orbit did not represent four unrelated source adapters. Nearly all storage reduction came with hard sharing, while the Mirror angles did not recover target-task quality and added substantial reconstruction compute.

### C — strongest counter-hypothesis

The source-task matrices were independent random maps and therefore poorly aligned for one conjugacy orbit. Structured related adapters or multiple physical bases might be compressible, but would need a distinct experiment and byte/compute accounting.

### U — unresolved

Natural NLP tasks, trained Transformer adapters, few-shot adaptation, sparse/low-rank bases, fused kernel implementation, and near-convergence remain untested. This result is a synthetic fixed-budget screen, not a claim about production AdapterFusion.

### Evidence labels

- **Fact:** measurements and serialized sizes/hashes in `results/development/` and `RESULTS_CORE.csv`.
- **Interpretation:** this Mirror bank failed for unrelated source functions; scalar codes offered no material advantage.
- **Hypothesis:** aligned or clustered adapters may admit compact Views, but independent/private components are needed for the tested task bank.

## Decision

FACT:
INTERPRETATION:
HYPOTHESIS:
BOUNDARY:
