# MA-877 status

- Status: FAIL (quality gates); replay verification incomplete
- Branch: research/ma-877-position-free-kv-transfer-20261111
- Base commit: 66aa8caa
- Protocol freeze: 79ff15fc
- Development/fresh settings freeze: 881b8992
- Fresh/audit opened: yes; rank 32 and disjoint prompt IDs frozen before evaluation
- Results: report a5d1552, core metrics a251487, fresh summary 9157725, provenance 454db49
- Verification record: c33e04d. Partial; see VERIFICATION.json. Exact fresh replay and final serializer test remain pending.
- Next P0 candidate: MA-878

## Decision summary

Position-free full independent maps improved on direct RoPE-space ridge in all three fresh seeds, but stayed +0.82/+2.97/+0.87 nat above native target prefill. Rank32 was +12.35/+16.44/+13.61 nat above native and 11.53/13.47/12.74 nat worse than independent content-space maps. Shared payload was 573,060 B versus 2,996,100 B independent (80.87% reduction); quality failed in all three worlds.

## Verification limitation

The execution environment became unavailable after the fresh run. Three algebra tests and model-cache roundtrip checks had passed before fresh evaluation; the deterministic fresh replay and final serializer buffer test could not be completed. Preserve the FAIL gate outcome but treat fresh values as not replay-verified.

## Blocker

Temporary execution-environment outage prevents final replay and local integrity checks. GPU was also unavailable for this CPU-only screen.
