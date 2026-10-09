# MA-257 — Factorized context superposition for held-out combinations

**H:** Separately bound procedure and domain factors may recover unseen task pairs with lower storage than per-task contexts.

**T:** Eight procedure and eight domain factors generate additive 256D vectors; 48 combinations train factor recovery and 16 are held out. Compare explicit tasks, native PA16 random context unbinding, Mirror factor-bank unbinding, plain factor table, and additive oracle. Fresh worlds 25710–25712 × seeds 0–2; actual serialized bytes charged.

**D:** Pending.

**C:** Plain factor tables may provide identical retrieval with simpler decoding; this known additive task generator favors factorization by construction.

**U:** Fresh unseen-pair error, bytes, runtime, and Mirror-specific advantage.

## Results

**D: FAIL for the registered Mirror hypothesis.** Fresh held-out NRMSE averaged 2.5414 for Mirror factor-bank unbinding, 6.9581 for native task-level PSP, 0 for generic factor table, 0 for additive oracle, and 0 for explicit full task bank. Mean serialized payloads were 20,765 B, 68,837 B, 50,857 B, 18,213 B, and 67,493 B respectively. Mirror compressed the payload relative to explicit tasks but destroyed functional accuracy; the plain factor table was exact and smaller than explicit vectors.

**Fact:** 9 fresh banks (worlds 25710–25712 × seeds 0–2), 45 result rows, actual torch-serialized payload sizes and SHA-256 recorded. Fresh pattern matches development.

**Interpretation:** Factorization of the known additive task family is useful, but this coordinatewise binding/unbinding implementation is not. There is no Mirror-specific advantage over the generic factor table.

**C:** The specific Hadamard-style one-shot unbinding has interference proportional to the number of factor addresses; this likely explains the large error.

**U:** Learned contexts, multi-replica/context cleanup, natural task vectors, and other binding operators remain untested. No broad parameter-superposition claim follows.
