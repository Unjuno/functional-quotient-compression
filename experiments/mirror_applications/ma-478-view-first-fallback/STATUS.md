# MA-478 status

- Status: FAIL (ordinary PCA + private residual reproduces Mirror exactly)
- Branch: `research/ma-478-view-first-private-fallback-20261008`
- Frozen protocol SHA-256: `349a046562b4224a16b969609f20418809317fc162333e0a4e7d4a2f19df42f9`
- Development seeds: 47801, 47802 (complete)
- Fresh/audit seeds 47811–47813: sealed, never accessed
- Serialized metric replay: exact

The residual gate triggered 16/256 private values, recovering the heldout off-basis behaviors. Native PCA plus the same fallback exactly matches the bank; the Mirror-specific gate fails.
