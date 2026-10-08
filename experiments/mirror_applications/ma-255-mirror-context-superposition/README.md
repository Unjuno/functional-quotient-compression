# MA-255 — Mirror context superposition vs Parameter Superposition

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`

## Hypothesis

H: For a fixed family of task-specific linear maps, a learned orthogonal Mirror context shared across the task bank can recover more held-out task behavior per serialized byte than fixed-context Parameter Superposition (PSP), naive task-code sharing, and a matched shared low-rank control.

## Mirror insertion

> **Mirror insertion:** this experiment adds a learned task coordinate `m_t` to one shared physical linear operator so that task-specific maps can be retrieved without storing a complete weight matrix per task.

- Native method before adding `m`: Parameter Superposition stores superposed task weights and uses fixed task context vectors to unbind them.
- Exact insertion point for `m`: task-indexed orthogonal transformation of a shared physical matrix; this screen uses a two-dimensional rotation chart with one learned angle per task and one learned shared base matrix.
- Persistent or dynamic `m`: persistent task code.
- Cheapest ordinary parameter that might provide the same freedom: task-code linear mixing of a shared low-rank basis.
- Physical object: a flattened task weight bank occupying one `T x P` tensor.
- Claimed logical multiplicity: T task-specific P-dimensional linear maps.

## Prior-art delta

- Closest prior art: PA16, Parameter Superposition (arXiv:1902.05522).
- PA16 establishes task-context binding/unbinding and context composition. This experiment tests whether a learned structured context coordinate improves the byte/quality/interference frontier over fixed PSP.
- Simpler controls: fixed random orthogonal PSP contexts, task-code shared rank-r basis, shared task-code factorization.

## Comparisons

1. Independent task maps (upper reference).
2. Native fixed-context PSP, using a fixed random row/column-factorized ±1 diagonal context and unbinding.
3. Naive task-code shared low-rank basis, with task coefficient vectors.
4. Learned orthogonal Mirror context: one shared base matrix, task angles, learned base angle.

All methods receive the same training pairs and optimization steps. Methods are compared at their measured actual serialized payload bytes and task-held-out test error. This post-fit representation screen does not claim matched-byte optimality if payloads differ.

## Gates

### PASS
At least 3/4 fresh seeds: Mirror test MSE <= 1.10x the best task-code/PSP control while using <= 75% of independent-model serialized bytes, with lower test MSE than both PSP and task-code controls at a byte ratio within 10%.

### FAIL
Mirror has >1.25x the best control test MSE on at least 3/4 fresh seeds, or it does not improve the byte-quality frontier over the simpler controls.

### NOT ESTABLISHED
Convergence failure, serialization mismatch, or development-to-fresh split contamination.

## Tuning boundary

Development seeds: 11, 23. Fresh seeds locked before execution: 101, 211, 307, 401. The screen uses fixed hyperparameters and direct fit/decomposition with zero optimizer updates; development can only select between the declared rank candidates {2, 4, 6}. Fresh results do not tune settings.

## Storage contract

The inference payload is a deterministic NPZ archive. Count all learned tensors, contexts, task codes, basis matrices and metadata in actual archive bytes. Report raw tensor bytes and archive bytes. Dataset-generation seeds and code are not inference state. Independent reference stores all task matrices.

## Compute contract

Report training examples, updates, a comparable dense-multiply MAC proxy, isolated wall time, and inference MAC proxy. The NumPy implementation is CPU-only; no GPU or language-model training is claimed.

## Decision

FACT: See `RESULTS_CORE.csv` and `VERIFICATION.json`.  
INTERPRETATION: Mechanism screen only.  
HYPOTHESIS: Learned task coordinates may improve useful task multiplicity per physical byte.  
BOUNDARY: Does not establish natural-language quality, general capacity, or superiority at converged fixed-byte frontier.

## Results — scoped mechanism evidence

FACT: On fresh seeds 101, 211, 307, 401, mean test MSE was 6.68e-17 for the 734-byte learned rotation View, 2.11e-30 for the 14,650-byte task-code rank-2 representation, 5.95 for 6,490-byte implemented PSP, and numerical zero for the 27,906-byte independent reference. The same pattern held on development seeds 11 and 23. Each method received 2,304 supervised training examples; optimizer updates were zero because this was a post-fit screen. Inference timing records only the dense output evaluation and excludes representation reconstruction; it is not a runtime comparison. The learned Mirror chart used 2.6% of independent payload bytes and 5.0% of task-code bytes.

INTERPRETATION: This supports a narrow byte-quality advantage for a chart that exactly matches the teacher's one-angle rotation family. PSP did not fit this structure under the implemented context family. The task-code control's gap arises because its rank-2 linear coefficient basis does not exactly encode the rotational orbit at this rank.

HYPOTHESIS: For task families with low-dimensional known orthogonal orbits, a task-addressed structured chart may compress useful functions more than an unstructured shared delta basis.

COUNTER-HYPOTHESIS: The result is induced by teacher/chart alignment and post-fit parameter extraction; a more suitable task-code basis, higher rank, or stronger PA16 context family could close the gap. This is not evidence of general Mirror-specific value.

UNCONFIRMED: training from examples, robust variation in the orbit family, matched-byte optimized baselines, full PA16 reproduction, LM quality, throughput at scale, and near-converged fixed-byte frontier.

Decision: PROMISING for this mechanism case only; no adoption claim.
