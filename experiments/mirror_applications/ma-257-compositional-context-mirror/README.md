# MA-257 — Factorized context superposition for held-out combinations

**H:** Separately bound procedure and domain factors may recover unseen task pairs with lower storage than per-task contexts.

**T:** Eight procedure and eight domain factors generate additive 256D vectors; 48 combinations train factor recovery and 16 are held out. Compare explicit tasks, native PA16 random context unbinding, Mirror factor-bank unbinding, plain factor table, and additive oracle. Fresh worlds 25710–25712 × seeds 0–2; actual serialized bytes charged.

**D:** Pending.

**C:** Plain factor tables may provide identical retrieval with simpler decoding; this known additive task generator favors factorization by construction.

**U:** Fresh unseen-pair error, bytes, runtime, and Mirror-specific advantage.
