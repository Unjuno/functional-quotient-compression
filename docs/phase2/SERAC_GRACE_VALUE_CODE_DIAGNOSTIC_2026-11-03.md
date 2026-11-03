# SERAC/GRACE shared value-code family diagnostic — 2026-11-03

## Trigger and scope

MA-636 and MA-637 both tested shared-basis compression of explicit per-edit counterfactual values. Both preserved the measured synthetic functions but the strongest native PCA value-code control exactly matched Mirror output/state, so no Mirror-specific effect was found. MA-638's GRACE key+View joint factorization is paused until a proposal differs from ordinary value PCA/low-rank coding.

This pause is limited to the shared value-code subfamily. It does not pause model editing globally; MA-639 tests LOKI's null-space-constrained edit coordinates.

## Evidence

- **MA-636 / SERAC:** 128 synthetic edit values; 96 training basis values, 32 held out. Scope hit 100%, false positives 0, held-out nRMSE ~0. Complete Mirror/PCA memory was 8,071–8,080 B vs 13,293–13,316 B explicit (~39% reduction), missing the 50% gate. Native PCA used identical arrays/bytes and lower measured latency.
- **MA-637 / GRACE:** 16/32/64/128 edit snapshots. Retention nRMSE ~0, scope hits 100%, false positives 0. Complete code memory savings grew from ~7% at 16 edits to ~38% at 128 edits; the frozen <=75% gate passed only at 64/128. Native PCA exactly matched arrays, bytes, retention and scope; decode added operations.

## Structural diagnosis

The tested compression gain came from ordinary PCA over values with a paid shared basis and per-edit codes. Routing keys/radii remained a large fixed cost at small bank sizes; basis reconstruction added compute. Mirror naming or coefficient coordinates did not change the function family or Pareto point relative to native PCA.

## Resume condition

Resume MA-638 only with a concrete basis/address operation not reducible to standard PCA, shared low-rank codes or ordinary key/value factorization. Reuse the same sequential snapshots, explicit memory control, scope metrics, serializer and latency accounting.
