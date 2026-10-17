# MA-1164 fused CPU follow-up — 2026-10-18

- Scope: **runtime engineering proof** of faster K×R Mirror execution and of exact posthoc learned-digit forward replay.
- Matrix-free structural Mirror: output equivalence PASS; PyTorch CPU eager forward latency reduction PASS in 16/16 grid cells; same-thread cached-dense beat in 12/16; other cases FAIL.
- Learned digits: 3.7× steady-state improvement at batch64 over old generated-weight PyTorch forward; **still ~37× slower** than native preexpanded MLP in the same FP64 CPU conditions.
- New tests 18/18 PASS, original microaudit 7/7 PASS. Randomized deterministic test fixtures and old previously-opened checkpoint only.
- GPU, full high-precision scalar optimization, real memory allocation tracing, backward/optimizer speed: UNTESTED.
- MA-1164 original Mirror-specific task quality/status remains historical FAIL. Do not change registry or main.
