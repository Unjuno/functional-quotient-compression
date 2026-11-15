# MA-1042 status

- Status: amended fresh audit pending
- Branch: `research/ma-1042-pgm-domain-residual-view-20261115`
- Official PGM-index source: pinned commit `c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407`; Apache-2.0
- Epsilon 8 remains frozen from development seeds 1042–1043
- Original held-out run on seeds 1044–1046 is preserved in `runs/exploratory_invalidated/` and excluded from confirmation
- Amendment AM-03 corrects one-key left-side predecessor refinement after the initial implementation missed rare absent-key cases
- New confirmatory fresh seeds: 1047–1049

## Current implementation

The pinned official DynamicPGMIndex applies and verifies insert/delete updates. As its iterators are forward-only, post-update predecessor queries use official static PGMIndex rebuilt from final keys. PGMIndex.search returns a guaranteed lower-bound interval `[lo, hi)`; exact predecessor correction searches from `max(0, lo-1)` so absent queries can return the preceding key. Mirror and ordinary affine coordinate controls use the same mapping and operations.

## Next action

Rerun corrected implementation on development seeds for correctness, then the frozen epsilon 8 on untouched confirmatory seeds 1047–1049.
