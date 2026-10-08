# MA-303 status

- Status: FAIL — Mirror-specific direct-code storage gate missed; held-out synthetic quality passed.
- Branch: `research/ma-303-factorized-layer-mask-mirror-20261008`
- Base commit: `7ce8e04`
- Fresh seeds 30311–30313 complete: Mirror 34,892B, direct factorized 34,920B, Piggyback 71,243B, independent masks 99,915B. Both factorized methods have held-out max nMSE 0 and no private fallback. Mirror is only 0.080% smaller than direct; FAIL for Mirror-specific gate.
- 15 packages hash/byte checked and metrics exactly replayed; four tests pass. Initial pre-amendment development remains preserved and excluded from corrected results.
