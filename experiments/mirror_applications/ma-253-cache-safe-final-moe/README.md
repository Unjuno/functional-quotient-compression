# MA-253 — cache-safe final-layer Mirror-MoE

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `01b515cf8d0603481cb00ff1d5be45541411eebb`

## Hypothesis

**H:** For domain-specific low-rank knowledge updates attached only to a decoder's final FFN, one shared matrix plus a short per-domain orthogonal view can approach independent final-layer LoRA quality with fewer serialized bytes while preserving exact prefix KV-cache reuse; placing the same view before a later attention layer should invalidate part of the cache.

## Prior-art delta

- PA14 (DMoE) uses independently updated knowledge experts and an uncertainty-aware router, attaching experts only at the final-layer FFN so prefix KV cache remains reusable.
- This experiment isolates expert storage with a supplied domain address, then checks the architecture-dependent cache property in a tiny causal decoder.
- Closest control: independent rank-2 per-domain LoRA updates on a shared base, matching DMoE's expert form and the synthetic teacher.
- Additional controls: hard-shared base, Mirror view plus rank-1 private residual, and independent full-rank private updates.
- No oracle fact answers are provided; the domain address is an explicit input to every method and is not treated as learned routing.

## Physical-to-logical claim

- Physical object: one final-FFN weight matrix shared across four domains.
- Mirror coordinate: four Givens angles per domain, applying a conjugate view of the shared matrix.
- Logical multiplicity: four domain-specific mappings without four full matrices.
- Cache property: final-FFN placement must leave all cached K/V projections invariant across domain addresses; moving the view to an earlier FFN should change later-layer K/V.
- Main risk: independent domain LoRA deltas may not be representable as coordinate changes; private parameters may be essential.

## Gates

### PASS

Across all three fresh worlds, final-layer Mirror MSE is within 10% of rank-2 DMoE LoRA and uses at least 20% fewer actual inference bytes; final-only view has zero max K/V cache difference across addresses; multi-layer view changes downstream K/V as predicted.

### FAIL

Mirror fails the quality/bytes gate while DMoE LoRA fits the task, or final-layer placement changes a cached K/V value. If a rank-1 private residual closes the gap, record that as evidence for a private-parameter threshold, not as Mirror-only success.

### NOT ESTABLISHED

Fresh data are incomplete, DMoE control does not learn, or cache probe is not deterministic.

## Tuning boundary

- Development world 25300; initialization 253000; two common learning-rate choices `{0.003, 0.01}`.
- Fresh worlds 25301, 25302, 25303, fixed before any fresh evaluation; independent target maps and samples.
- Fixed architecture, rank, update count, seeds, cache test, and thresholds are in `PROTOCOL.json`.

## Storage and compute

Actual serialized `torch.save` state-dict and config bytes are authoritative. Count the shared base matrix, all adapter/view/private tensors, model metadata, and routers if present. The mechanism task supplies a domain address and therefore does not measure router bytes or routing compute. Report examples, updates, MAC proxy and wall time. Cache audit reports exact K/V max absolute difference.

## H / T / D / C / U

H is stated above. T, D, C and U will be added after frozen fresh evaluation. This is a synthetic parameter-injection mechanism screen, not a natural-language benchmark or evidence for external expert loading performance.
