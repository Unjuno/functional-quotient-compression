# MA-1063 status

Phase: completed, scoped FAIL. Frozen audit used two fresh training seeds over 99,718 MovieLens 1M audit rows. Serialized replay passed for all 12 method/seed payloads. Mirror meets the full-table storage and no-view throughput gates, but fails the required quality margin against controls and misses the full-table AUC tolerance. It is slightly behind the matched diagonal. See `RESULTS.md` and `VERIFICATION.json`.

Next eligible registry candidate: MA-1064, DHE dynamic rare-ID private fallback (PA338). This candidate tests selective per-ID private state and will use a separate branch/protocol.
