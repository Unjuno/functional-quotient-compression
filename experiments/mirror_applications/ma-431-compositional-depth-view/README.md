# MA-431 — compositional Mirror views over recurrent depth

Status: SCREENING. Evidence lane: MECHANISM / STORAGE / RUNTIME.
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`.
Prior art: PA72, Universal Transformer.

## H — falsifiable hypothesis

For sequences of noncommuting depth operations, a shared recurrent block with a compact Mirror coordinate at each step will recover held-out ordered compositions with fewer serialized bytes than independent step matrices, and will improve the quality/byte frontier over a shared block conditioned by a simple low-rank gate.

## Mirror insertion

> **Mirror insertion:** this experiment adds a scalar angle coordinate `m_t` to the shared recurrent transition through a Givens-conjugated view `R(m_t) W R(-m_t)`, so different ordered logical step operators can be applied without storing a separate full matrix at every depth.

- Native method: Universal-Transformer-style repeated shared transition, with timestep conditioning.
- Exact insertion: conjugate one shared square transition by a deterministic two-dimensional Givens rotation selected by `m_t`.
- `m_t` is an explicit per-step address; the fixed angle codebook is deterministic from the operation ID and fully described in the payload metadata.
- Cheapest ordinary control: a rank-one, low-rank-conditioned residual/gate over the same shared transition.

## Physical-to-logical claim

- Shared object: one learned transition matrix `W` reused at each recurrent step.
- Mirror coordinate: one scalar Givens angle per operation symbol, plus the step's symbol ID.
- Logical multiplicity: ordered compositions of two noncommuting operation views.
- The candidate can save storage when the same few operation views recur across long programs.
- It may fail because timestep conditioning, a rank-one gate, or simple independent per-symbol matrices can express the same task with equal or lower runtime.

## Comparisons

1. Static tied block with no step coordinate.
2. Universal Transformer control: shared transition with learned depth embedding and a shared low-rank FiLM/gate.
3. Generic rank-one residual/gate control with the same address bits.
4. Mirror Givens-conjugated shared transition.
5. Independent per-symbol matrices as the matched-quality upper reference.

Inputs provide the same operation-symbol sequence to every method. Training uses lengths 1–3; the primary held-out score is length 6 and unseen ordered programs. The task teacher uses two noncommuting 2D shears embedded in a 16D state; targets are exact sequential applications to held-out inputs.

## Gates

### PASS (mechanism screen only)

On every fresh world, Mirror must have length-6 relative MSE no more than 1.05× the independent per-symbol reference, and either (a) use at least 20% fewer actual payload bytes than that reference or (b) improve MSE by at least 10% over the byte-near low-rank gate control (within 10% payload bytes) at no more than 1.10× its active MAC proxy.

### FAIL

On development, Mirror misses the quality gate, uses more bytes than the independent reference without a quality gain, or is matched/dominated by the simple low-rank gate at byte-near cost. A dev FAIL seals the fresh worlds.

### NOT ESTABLISHED

Invalid split, metric, serialization, or execution accounting; materially divergent results across fresh worlds; or inability to reproduce the registered control.

## Tuning boundary

- Development worlds: 43100–43101, seeds 0–2.
- Only development may select learning rate from `{1e-3, 3e-3}`, coordinate scale from `{0.5, 1.0}`, and model width from the fixed registered 16D value (no width changes).
- Fresh worlds: 43110–43112, seeds 0–2, locked before development; opened only if the full development gate passes.
- Fresh data are never used for model or protocol changes.

## Storage and compute contract

Serialize all learned weights, learned depth embeddings, per-symbol codes, codebook/angle metadata, and configuration in one inference payload. Report actual `torch.save` bytes. Derive MAC proxy from executed matrix operations and report training examples, updates, train wall-clock, single-thread CPU inference wall-clock, and examples/second. No optimizer state is an inference object.

## Boundaries

This is a synthetic mechanism screen, not a language-model or natural-sequence capacity claim. Operation IDs are supplied to all methods; routing is oracle/symbol-conditioned and not learned. Combination count alone is not capacity. Any result must be followed by a task-specific natural-data experiment before making an application claim.
