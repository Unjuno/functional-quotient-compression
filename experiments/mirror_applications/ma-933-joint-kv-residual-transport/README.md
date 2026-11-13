# MA-933 — joint residual and KV cross-model chart

Status: **FAIL (scoped TinyStories mechanism screen)**. A rank-8 source/target chart shared across residual, key and value tensors reduced connector bytes by 97.3%, but missed the frozen next-token quality gate on all three seeds: +3.90 nat/token versus +1.14 for independent affine. Source model weights were included in full deployment accounting.

See [RESULTS.md](RESULTS.md), frozen [PROTOCOL.json](PROTOCOL.json), [AMENDMENTS.md](AMENDMENTS.md), and [VERIFICATION.json](VERIFICATION.json). This is not a fused serving benchmark or PA236 reproduction.
