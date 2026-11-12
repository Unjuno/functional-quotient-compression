# MA-889 status

- Status: FAIL
- Branch: `research/ma-889-confidence-cache-fallback-20261112`
- Protocol and source-only thresholds were frozen before fresh scoring.
- Fresh: seeds 12889/12893/12899, three disjoint article buckets.
- Quality: Pythia-160M used 100% fallback; SmolLM2 mean regret +1.277 nat and failed in 2/3 seeds.
- Fanout latency beat native in 3/3 buckets; payload is 7,787,782 B, only 30 B above full ridge maps.
- Verification: six tests passed; serialized NLL replay delta 0.0. The post-score MA889 magic-header correction changed no model state or metrics; old/new hashes are recorded.
- Next eligible P0: MA-891.
