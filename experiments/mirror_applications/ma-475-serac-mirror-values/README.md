# MA-475 — SERAC external memory values with Mirror code compression

Status: **PROMISING for aligned memory-value storage; no Mirror-specific performance claim**
Evidence lane: memory value storage / retrieval / quality / latency
Branch: `research/ma-475-serac-mirror-values-20261009`
Protocol frozen: `510715fb`; fresh worlds 47510–47512 × seeds 0–2.

## H — Hypothesis

Holding SERAC's key memory and scope retrieval fixed, a shared value basis plus a per-edit angle can store many aligned counterfactual values more compactly than explicit vectors while preserving hit behavior, paraphrase response and deferral on unrelated inputs.

## T — Test

We created an analytic 16D key/value memory with 64 entries per world. Keys were random unit vectors. Values had fixed norm 1.2 on a known shared 2D plane. All methods used the same nearest-cosine retrieval and fixed scope threshold 0.90. Queries included exact keys, four noisy paraphrases per key (Gaussian sigma 0.03), and 256 unrelated random vectors. Controls were no edit, SERAC explicit key/value memory, generic Cartesian value coefficients over the same shared plane, and Mirror angle codes. Actual `torch.save` payloads included keys, shared bases, code/value state, IDs, scope threshold and metadata. Metrics separate retrieval recall, value error, unrelated false triggers, bytes, lookup wall time and a MAC proxy for value decoding.

## D — PROMISING for memory-value storage only

**Facts:** On all nine fresh world/seed pairs, SERAC, generic coefficients and Mirror had 100% exact and paraphrase hit recall, zero false triggers on unrelated queries, and value NRMSE below 1.3e-7. At N=64, explicit SERAC used 10,529B, generic coefficients 7,261B, and Mirror 7,069B. Mirror was 67.1% of SERAC bytes and 97.4% of the generic coefficient payload. At N=8 Mirror and generic coefficients both used 3,037B; at N=1 Mirror and generic both used 2,589B, larger than the 2,273B explicit memory. Median measured lookup time at N=64 was 0.56 microseconds/query for Mirror, 0.57 for generic coefficients, and 0.59 for SERAC on this CPU microbenchmark; this times key search and retrieval after value decoding.

**Interpretation:** Shared values explain most of the storage reduction: generic coefficients already cut 31% from SERAC. The angle code saves a further 192B (2.6%) at N=64 over Cartesian coefficients and is tied at N=8. This is a small fixed-norm coding gain, not new edit scope or better retrieval. Value decoding adds work, and measured lookup excludes its one-time bank reconstruction.

## C — Strongest counter-hypothesis

The values were deliberately placed on a known fixed-norm circle; ordinary polar coding gives the same code. Real SERAC counterfactual functions need not share this geometry, and a shared value basis can fail even while the scope classifier remains perfect. Most bytes saved came from generic shared-basis coding, not Mirror.

## U — Unknown

Natural SERAC edits, learned basis discovery, nonlinear counterfactual models, value decode/load costs in a deployed serving system, false triggers under distribution shift, and lifelong memory growth are untested. The exact synthetic classifier does not reproduce SERAC's learned scope classifier.

## Decision

**FACT:** On the aligned screen, compressed value banks preserved identical retrieval outcomes and lowered bytes at N=64; generic coefficients nearly matched Mirror.
**INTERPRETATION:** Fixed-norm shared values can reduce external-memory state, but the incremental Mirror gain is only 2.6% over a simple shared-basis control.
**HYPOTHESIS:** Natural counterfactual outputs may contain reusable low-dimensional value structure, but this experiment does not establish that.
**BOUNDARY:** Synthetic memory and hand-specified value orbit; no factual language, learned SERAC, or deployment claim.
