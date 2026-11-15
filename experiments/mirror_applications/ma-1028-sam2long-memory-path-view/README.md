# MA-1028 — SAM2Long memory-path feature views

**Registered decision: NOT ESTABLISHED.** The required two-object SAM2Long path-tree control cannot be executed with the official implementation as supplied. A single-object run is recorded as exploratory only.

## H — Hypothesis

Three SAM2Long path memories may be stored as one shared memory base plus two per-path Givens angles without losing useful tree diversity. The registered test requires >=25% lower complete-session bytes, <=1 pp J&F loss against independent pathways, and a >=1 pp Mirror advantage over rank-1 additive in at least two of three fresh clips.

## T — What ran

We obtained official SAM2Long commit `d70b50a7936fec55af20123530e92e794ad901a4` and the official SAM2.1 Hiera-tiny checkpoint. A 256px feasibility run failed before scoring because the official relative-attention source assumes 4096 spatial tokens. Before any quality metric was inspected, the protocol was amended to upstream 1024px resolution; the 3-frame CPU smoke succeeded.

The scored exploratory run then tracked one object over one 12-frame synthetic occlusion/reappearance clip using the official three-path tree. Independent, hard-shared, two-angle Givens Mirror, and rank-1 additive memories were compared. The frozen registered protocol required two objects. The official path-selection code uses scalar `object_score_logits.item()` and comments that its path logic assumes one object, so the registered two-object control was not reproducible. No two-object audit result is claimed.

## D — Decision

**NOT ESTABLISHED** for MA-1028. The one-object measurements below diagnose the path-memory mechanism but do not satisfy the registered object count or three-seed protocol.

| One-object exploratory method | J&F | path memory bytes | full session bytes | FPS, single CPU run |
|---|---:|---:|---:|---:|
| Independent | 0.844057 | 16,777,216 | 337,462,081 | 0.0809 |
| Hard shared | 0.844057 | 6,291,456 | 326,977,431 | 0.0831 |
| Givens Mirror | 0.844057 | 6,291,696 | 326,979,305 | 0.0863 |
| Rank-1 additive | 0.844057 | 11,534,396 | 332,223,578 | 0.0903 |

All four selected prediction masks are byte-identical on this clip. Mirror reduces the targeted path-memory state by 62.50%, but complete-session savings are 3.11%; even deleting the entire independent path-memory bank could save only 4.97% of that one-object session. Mirror uses 240 B more than hard sharing. Single-run CPU timing is not a throughput claim.

## C — Strongest counter-hypothesis

The tree selects identical useful masks regardless of memory coding in this simple synthetic clip; the unchanged image cache, model and per-path prediction outputs dominate the complete session. A harder natural long-video sequence might expose path diversity, but the current official implementation cannot run the registered two-object tree.

## U — What remains unknown

Two-object path interactions, fresh-seed quality, natural-video/benchmark quality, longer horizons, GPU throughput, and learned memory coordinates were not established. The source-level multi-object limitation blocks the frozen registered comparison.

## Fact / interpretation / hypothesis

- **Fact:** the official 1024px one-object three-path smoke and 12-frame comparison ran; all four selected masks match exactly; serialized payload hashes and byte counts were captured; the frozen protocol specified two objects.
- **Interpretation:** the one-object screen shows no Mirror-specific selected-mask benefit and little complete-session compression, but it cannot decide the registered two-object hypothesis.
- **Hypothesis:** a vectorized object-aware SAM2Long path scorer could permit the registered experiment; that would require a new official-control implementation and a new execution amendment.

Detailed blocker and evidence: `EXECUTION_BLOCKER.md`, `runs/exploratory_one_object.json`, and `runs/exploratory_session_headers.json`.
