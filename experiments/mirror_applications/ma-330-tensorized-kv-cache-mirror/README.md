# MA-330 — Tensorized KV cache/view reconstruction

Status: PROMISING for aligned shared-cache representation; no decode-speed gain established. Dedicated branch: `research/ma-330-tensorized-kv-cache-mirror-20261008`.

## H — hypothesis

A rank-4 temporal basis plus compact layer/head coordinates can represent multiple logical K/V cache views while preserving held-out causal attention outputs. Private state may be needed for unrelated layers, and reconstruction may offset cache savings at decode time.

## T — task and execution

Synthetic causal attention cache: 4 layers, 4 KV heads, 128 tokens, feature width 16. Three layers are orthogonal rotations of shared per-head K/V states. The fourth layer is deliberately unrelated. Canonical K/V states use rank-4 temporal bases; Mirror rotates the reconstructed K/V state using one phase per layer/head. Controls: independent full FP16 caches, MLKV-style hard sharing, shared factors plus direct cos/sin coefficients and a private cache for the unrelated layer, Mirror phase plus the same private fallback, and Mirror without private state. Two development seeds and three fresh seeds were run. No optimizer updates; synthetic task coordinates are oracle-known. Fresh outputs were evaluated after loading the actual serialized FP16 payload. Attention output nMSE, per-layer error, payload bytes, reconstruction MACs, wall time, and query throughput were recorded.

The frozen protocol included an amendment before fresh access: the rank-4 temporal state representation, per-layer error, and separate reconstruction/attention timing were fixed. Gates/seeds stayed unchanged.

## D — decision

**PROMISING for aligned shared-cache views, narrowly scoped.** In all three fresh worlds, Mirror payload was 36,458B and attention-output nMSE was 5.15e-8–6.39e-8. Independent cache was 131,752B, so Mirror reduced actual bytes by 72.3%. The direct cos/sin control used 36,488B and had comparable near-zero output error; Mirror saved 30B (0.082%) total. The phase coordinate halves the view-code values (one FP16 phase vs FP16 cos/sin pair per aligned role), but shared basis, coefficients, and private cache dominate total storage. This is not a material end-to-end storage advantage over the simple control and is not ADOPTED.

The no-private Mirror state was 3,238B but had aggregate output nMSE 0.919–0.985; per-layer results show aligned layers around 1e-7 and the unrelated fourth layer around 1.0. A private cache for that layer raised payload to 36,458B and restored quality. Decode query throughput did not show a consistent advantage over direct coefficients; eager CPU timing is a small synthetic screen.

## C — strongest counter-hypothesis

The main storage reduction comes from sharing the rank-4 temporal basis and canonical K/V state, a standard low-rank/tensor sharing effect. Direct sin/cos coefficients recover the same functions at almost the same total bytes. The tiny 30B phase-code saving could be an archive/layout artifact at this scale, and oracle-known coordinates omit the cost of discovering them.

## U — unresolved

Natural-language NLL and trained Transformer KV caches are untested. Phase coordinates were supplied by the planted teacher rather than learned from task data. The CPU screen is too small to establish a deployment latency improvement. Larger layer/head banks may amortize fixed basis/header costs differently.

## Fact / interpretation / hypothesis

**Fact:** 15 fresh FP16 payloads were checked for exact bytes and SHA-256; reloaded attention output nMSE replayed exactly. Five tests pass. Mirror is 72.3% smaller than independent, and only 30B smaller than direct cos/sin. No-private quality fails only on the unrelated layer.

**Interpretation:** One physical temporal K/V basis can support several distinct logical layer views when they share a known rotation family. Unrelated cache functions cross the private-state boundary. A simple direct control reproduces nearly all Mirror storage/quality behavior.

**Hypothesis:** A learned phase code may be useful for much larger aligned layer/head banks, but that amortization and learning-cost claim remains untested.

## Evidence files

- `RESULTS_CORE.csv`: fresh summary rows.
- `artifacts/fresh/`: serialized inference payloads and metrics records.
- `source/verify.py`: byte/hash and FP16-reloaded output replay.
- `VERIFICATION.json`: verification record.

The initial `artifacts/development/` folder is retained as exploratory evidence from the pre-rank-4 implementation and is excluded from comparisons. `artifacts/development_frozen/` contains the final development implementation used to freeze source/protocol before fresh seeds.
