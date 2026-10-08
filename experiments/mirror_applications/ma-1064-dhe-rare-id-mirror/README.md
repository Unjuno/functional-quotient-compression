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

## Result

**FAIL for this frozen screen.** Across three audit seeds, DHE + Mirror angle + rank-1 residual reached rare-item AUC **0.44456**, only **0.00078** above the byte-near ordinary rank-2 residual control (0.44378), versus the required +0.01. Overall AUC was 0.49622 versus 0.49617 for rank-2 residual. The candidate serialized to **189,240 B**, 344 B larger than that control and 55.6% of the 340,632 B full table. Its batch-1 P95 was **0.200 ms**, about 2.2× DHE's 0.090 ms, failing the 1.2× latency gate.

All learned models performed poorly on this temporal split: candidate audit logloss averaged 0.86783, while a constant train-prevalence predictor scored 0.68478; full lookup AUC was 0.506. For seed 31, DHE train AUC/logloss were 0.695/0.625 versus dev 0.537/0.799; full-table train was 0.814/0.529 versus dev 0.522/0.851. This points to strong temporal generalization failure in this small setup. Only 512 audit events belonged to the train-rare stratum. Treat this as a negative result for the implemented screen, not a general DHE or Mirror conclusion.

Amendment 1 corrected only method metadata in the serialized payload after discovering a Mirror descriptor on non-Mirror controls. Old metrics are preserved in `AMENDMENT1_PRE_CORRECTION.*`; all 18 audit prediction metrics replayed identically, and only corrected method-specific payload byte counts are used above. No model weights, settings, split, seeds or gates changed.
