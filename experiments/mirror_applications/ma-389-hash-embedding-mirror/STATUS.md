# MA-389 status

**FAIL — native Hash Embedding quality/collision gates missed; fresh seeds sealed.**

- 2026-10-08: Protocol frozen from PA58 before development. Compared full table, native Hash Embeddings, unweighted hash, scalar importance, and Mirror angle codes.
- Two development seeds used identical within-world labels and hash indices across methods. Native hash reached 0.9643/0.9495 accuracy; Mirror reached 0.9130/0.9083.
- Mirror payload was 7,871 B vs 9,929 B native (79.2%), and it substantially beat scalar, but failed the 2-point quality margin and populated collision-degree limits.
- Ten payloads reloaded and replayed metrics; 4 tests passed. Fresh seeds 38911–38913 remain unopened.
- Continue with MA-391, quotient/remainder compositional embeddings.
