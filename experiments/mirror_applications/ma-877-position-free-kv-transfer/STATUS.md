# MA-877 status

- Status: FAIL (quality gates); replay reproduces gate verdicts with small CPU numeric variation
- Branch: research/ma-877-position-free-kv-transfer-20261111
- Base commit: 66aa8caa
- Protocol freeze: 79ff15fc
- Development/fresh settings freeze: 881b8992
- Fresh/audit opened: yes; rank 32 and disjoint prompt IDs frozen before evaluation
- Results commit: `b3ff4f864` (report, metrics, full replay and provenance)
- Verification record: follow-up commit pending. Four tests pass; fresh prompt IDs, serialized payload sizes and all gate verdicts reproduce. NLL max deviation is 0.02051181 nat. See VERIFICATION.json.
- Next P0 candidate: MA-878

## Decision summary

Position-free full independent maps improved on direct RoPE-space ridge in all three fresh seeds, but stayed +0.82/+2.97/+0.87 nat above native target prefill. Rank32 was +12.35/+16.44/+13.61 nat above native and 11.53/13.47/12.74 nat worse than independent content-space maps. Actual serialized payload was 573,125 B versus 2,996,165 B independent (80.871% reduction), including header and pair metadata; quality failed in all three worlds.

## Replay note

The fresh re-evaluation reproduces every per-seed gate verdict and the actual serialized bytes. Numeric NLLs vary by up to 0.02051181 nat across CPU runs; this is recorded rather than described as exact numeric replay.

## Blocker

GPU was unavailable for this CPU-only screen. The integrity checker passed on the cumulative branch after the registry-row repair.
