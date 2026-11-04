# MA-653 — Titans long-term Mirror memory

Status: SCREENING. Separate short-term cache from surprise-gated writable long-term memory and sweep private residual rank.

The synthetic task stores 64 values behind one-hot keys, keeps the newest eight in an exact short-term cache, and updates a persistent linear long-term memory only when surprise exceeds a threshold. Compare full matrix memory, shared rank-4 basis codes with private residual ranks 0/2/4/8, native low-rank controls and an explicit key-value table. Every state component is serialized and charged.

This tests a Titans-inspired memory mechanism, not the full paper architecture. See `PROTOCOL.json`.
