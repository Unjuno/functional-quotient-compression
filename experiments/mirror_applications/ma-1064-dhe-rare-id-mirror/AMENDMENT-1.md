# Amendment 1 — inference metadata labels

Date: 2026-10-08 UTC

## Reason

The first post-audit serialization placed a Mirror-plane metadata string on every method, including full-table and plain-DHE controls. This did not change model weights, training, development checkpoint selection, audit predictions, optimizer updates, timing, or frozen gates. It slightly overcharged the non-Mirror controls' actual payloads.

## Frozen boundary

Keep the original protocol, all 18 selected checkpoints, audit split, seeds, optimizer settings and quality metrics unchanged. Re-serialize the same checkpoints with method-accurate inference metadata. Preserve the original files as `AMENDMENT1_PRE_CORRECTION.csv` and `.json`; exclude only their byte counts and whole-payload hashes from the final byte frontier. The amended payloads still use actual safetensors bytes including their complete metadata.

No audit values were used to choose or modify a model, seed, split, parameter or gate. This amendment is a serializer-accounting correction only.
