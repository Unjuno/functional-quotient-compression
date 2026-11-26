# MA-1062 — DLRM shared embedding table with feature-field Views

**Mirror insertion:** learn eight paired-plane angles for each of four embedding fields after a single H=128 or H=512 field-salted hash table lookup. The DLRM dense bottom/top network is the same compact architecture in every method.

H tests whether the field rotations give useful distinct logical embeddings from one physical table. Required comparisons include plain shared hashing, equal-row per-field hashing, paired and per-coordinate diagonal controls, and independent full-cardinality tables. Success gates and data split are frozen in `PROTOCOL.json` before development. Official MovieLens terms are retained there; the archive stays outside Git.

Fresh audit uses official `ua.test` only after `FRESH_FREEZE.json` is committed. This is explicit rating preference prediction, not CTR. Actual serialized payload bytes and embedding lookup latency are required metrics.
