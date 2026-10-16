# DMFFN-003 status (2026-10-17)

- Experiment: Natural learned FFN transfer and factorized SwiGLU Mirror compression.
- Source and four stage-specific protocols frozen and executed in container CPU; 120 student model fits + 4 x9 teacher models.
- A: mirror/gain heldout narrow gate PASS 2/3; BUT mirror/tied heldout FAIL 0/3.
- B: direct real label heldout useful transfer FAIL 0/3.
- C: full nine-label quality/byte vs independent and cheap readout FAIL 0/3.
- D: 6k-update same-quality long-budget FAIL 0/2; validation early stopping needed.
- Storage: model-only mirror 20,489 B versus independent 153,685 B; with identically charged 2,917 B preprocessor, full inference state 23,406 B versus 156,602 B (85.05% smaller), not equal quality.
- Verification: 18 unit tests, 120 inference payloads reload/hash/metric pass, four deterministic student retrainings plus one independent-teacher parameter check.
- Scope: natural digits, manually composed labels, no actual Transformer/LLM or GPU result.
- No modification to previous DMFFN-001 or main; research-only branch.
