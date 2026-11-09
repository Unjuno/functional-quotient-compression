# MA-701 accounting amendment

The initial audit payload serialized block-diagonal control matrices in dense D×D arrays, charging deterministic zeros. The first source/results are preserved under `source/attempts/dense_block_serialization/` and are invalid for the block-control byte comparison.

The corrected run stores only each 8×8 block plus shape metadata, which is the actual reconstructable inference payload. No transform, target, seed, quality gate, or candidate setting changed. The same frozen development and audit seeds are replayed; no selection or tuning decision was made from the initial accounting run.
