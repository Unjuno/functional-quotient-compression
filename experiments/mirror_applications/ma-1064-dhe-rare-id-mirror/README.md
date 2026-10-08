# MA-1064 — DHE rare-ID Mirror fallback

Status: SCREENING
Evidence lane: QUALITY / STORAGE / RUNTIME
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`

## H — falsifiable hypothesis

On a temporal MovieLens-100K audit, adding a learned rare-item Givens View coordinate and rank-1 private residual to a deterministic multi-hash embedding model will improve rare-item AUC by at least 0.01 over a byte-near rank-2 ordinary residual control, while keeping overall AUC within 0.003, logloss within 0.01, serialized bytes below 80% of a full item table, and P95 lookup latency below 1.2× native DHE.

## Draw25

MA-1064 was sampled uniformly from 529 eligible P0/UNTESTED candidates after excluding live remote MA branches and existing experiment directories. Pool hash, 256-bit seed, index 459 and exact exclusions are frozen in `source/draw25_exclusions.json`.

## T — data and controls

Use MovieLens-100K binary feedback (`rating >= 4`) with a global timestamp split of 70% train, 10% development, 20% fresh audit. Rare IDs are defined using train counts only (`1–5`), cold IDs have zero train events, and common IDs have six or more. Train three model seeds; choose each checkpoint by development logloss without changing frozen architecture or optimizer settings.

Compare: full item table; deterministic four-hash DHE-style decoder; DHE plus rare-item Mirror angle only; DHE plus rare-item rank-2 additive residual; DHE plus Mirror angle and rank-1 private residual; and DHE plus a full private residual for rare items. Mirror angle and residual use separate charged parameters. Every model includes its complete user table, learned weights, rare-ID list, hash metadata and decoder state in serialized-byte accounting.

The DHE implementation follows the published multi-hash-plus-decoder mechanism, with a small fixed configuration suitable for this dataset. It is not the paper's optimized production kernel. Results will not claim production DHE throughput.

## D — gates

**PASS for this screening hypothesis** requires all of: mean rare-item AUC gain ≥0.01 over rank-2 residual control with positive gain in at least 2/3 seeds; overall AUC drop ≤0.003; logloss increase ≤0.01; payload ≤80% of full lookup; P95 latency ≤1.2× DHE; and the Mirror candidate is not dominated by the byte-near ordinary residual control.

**FAIL** if any gate fails. **NOT ESTABLISHED** if the public data/protocol or implementation cannot support a valid comparison. No parameter-capacity claim follows from fixed-training-budget results.

## C — strongest counter-hypothesis

DHE's deterministic hash features already share useful structure; rare-item gains come from ordinary residual parameters or frequency-aware table allocation, while the Mirror code adds state and lookup work without a distinct quality/byte advantage.

## U — boundaries

This is one small public recommendation dataset with implicit binary labels. It does not establish large-scale recommendation behavior, production GPU throughput, cold-start quality beyond MovieLens, or superiority to optimized TT-Rec/Punica serving kernels. The exact DHE paper implementation is not reproduced.

See `PROTOCOL.json`, `STATUS.md`, `RESULTS_CORE.csv`, and `VERIFICATION.json`.
