# MA-581 results

## Fact

Shared rank-128 MLA PCA increased next-token NLL versus FP16 by 1.1219 nat/token in seed 58101 and 0.2741 in seed 58102. Adding rank-4 shared residual head codes changed those deltas to +1.0364 and +0.4116: a 0.0855-nat improvement over MLA in seed 1, but a 0.1375-nat regression in seed 2. Private per-head rank-4 residual bases scored +1.0370/+0.2444; they are slightly better than the shared residual method in seed 2.

Actual state and one-session bytes: FP16 cache 787,214 B; MLA rank128 model basis 1,586,216 B plus 99,086 B/session; Mirror shared residual model basis 1,590,068 B plus 148,530 B/session; private-head residual bases 1,636,152 B plus 148,530 B/session. For eight active caches, Mirror totals 2,778,308 B versus 6,297,712 B for eight FP16 caches (44.1%). For one active cache, Mirror totals 1,738,598 B, above the 787,214 B FP16 cache. The shared Mirror and native residual session arrays are exact matches.

Both Mirror/native use the same shared residual dictionary state. Total deployment improvement at eight sessions comes from ordinary PCA latent caching, not a distinct Mirror mechanism. File sizes and SHA-256 values for each basis, session payload and metrics file are recorded in `ARTIFACT_PROVENANCE.json`.

**T:** two Pythia-70M/WikiText-2 valid worlds. Four 64-token train prefixes fit a centered per-layer joint K/V PCA basis of rank 128 and rank-4 residual dictionaries. Sixteen one-token queries per seed score reconstructed caches. Controls: FP16, rank128 MLA, shared residual rank4 Mirror, the exact native shared residual dictionary, and private per-head residual bases. Fresh test seeds 58111–13 remain sealed.

## Interpretation

MA-581 FAILS its frozen quality gate: Mirror NLL delta is +1.036/+0.412 nat/token, versus a maximum allowed +0.05. The residual improves MLA in one world and harms it in the other. Its exact native shared-dictionary alias means the residual code does not show Mirror-specific value. The eight-session storage frontier is promising for generic MLA compression, but one-session storage is worse than FP16 after charging decoder state.

## H / T / D / C / U

**H:** a shared rank-128 latent plus rank-4 head/KV residual codes can approach FP16 cache quality with lower deployment bytes than per-role cache state, beyond the native shared-dictionary control.

**D:** FAIL. NLL misses by 0.986/0.362 nat/token; the shared residual code exactly aliases the native dictionary and crosses the MLA-only baseline in opposite directions across seeds.

**C:** rank128 joint PCA discards too much cache information; a rank-4 shared residual dictionary is too small and is ordinary shared-basis coefficient coding.

**U:** full-sequence perplexity, higher latent/residual ranks, longer contexts, learned MLA projections that replace rather than follow Pythia K/V generation, serving kernels, and fresh test transfer.
