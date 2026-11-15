# MA-1031 development record

Development seed 103100 only. Audit worlds 103101–103103 have not been generated.

| Method | Mean J&F | Clicks/video | Click-history bytes | Full-session bytes | FPS | ID switches |
|---|---:|---:|---:|---:|---:|---:|
| no_correction | 0.640867 | 0 | 119 | 169,981,625 | 2.927 | 0 |
| direct_clicks | 0.843256 | 40 | 544 | 170,346,945 | 1.724 | 0 |
| grid8_clicks | 0.846133 | 40 | 318 | 170,346,689 | 1.948 | 0 |
| mirror_polar | 0.800350 | 40 | 503 | 170,346,945 | 2.005 | 4 |
| full_mask_upper | 0.949184 | 0 | 121 | 175,595,993 | 2.595 | 0 |

The per-object Mirror half-radii are frozen in `runs/frozen_settings.json`: 15.6797, 34.7188, 25.4531, 16.4531.

Development-only observation: Mirror uses 503 B vs direct float32 correction history at 544 B, a 7.5% reduction (below the 25% gate). The native 8 px grid uses 318 B and has slightly higher J&F than direct clicks (.84613 vs .84326), dominating Mirror on both history bytes and quality (.80035). Mirror and direct have equal full-session bytes at 170,346,945 B. These seed results are retained; the threshold/radius is frozen and audit remains unopened.

The source correction fixed point-pair axis ordering in the Mirror decoder after the initial dev attempt stopped before its Mirror score; all methods were rerun. Full resumable states are serialized in memory and exact lengths/SHA-256 retained, while duplicate large state binaries are not committed; history payloads and predictions remain available. These were pre-audit implementation/accounting changes only; no quality gates, click rules, coding, or frozen radii changed. No audit-world data or labels were accessed.
