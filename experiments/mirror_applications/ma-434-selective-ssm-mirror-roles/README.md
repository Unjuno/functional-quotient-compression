# MA-434 — selective SSM role views

Status: SCREENING; PA73 reviewed, protocol frozen before runs.

## H — Hypothesis
One shared input-selective diagonal SSM plus a scalar role view over log-decay can reproduce multiple recurrent functions with lower actual bytes than storing each full SSM.

## T — Frozen mechanism screen
A 16-state Mamba-like recurrence uses token-dependent delta, B_t and C_t, while role code m shifts the shared log-decay vector. Twelve codes, 32 sequences/role, length 256. Compare one shared SSM, native role-conditioned decay bias, and independent full per-role copies. This is a fixed-parameter synthetic mechanism/storage/runtime screen, not a trained Mamba or language-model experiment. Actual compressed payload bytes include all role state and metadata. Fresh seeds 43411–43413 remain sealed.

See `PROTOCOL.json` for exact recurrence, seeds, metrics and gates.
