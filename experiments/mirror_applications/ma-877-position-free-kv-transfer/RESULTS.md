# MA-877 — position-free KV transfer after inverse RoPE

Status: **FAIL** on the frozen quality/storage frontier; exact replay verification is incomplete because the execution environment disconnected after fresh evaluation.

## H — Hypothesis

For the frozen Pythia-70M→SmolLM2-135M pair, fitting affine KV maps after removing source/target RoPE and reapplying target RoPE should improve held-out NLL over direct RoPE-space ridge. A rank-32 shared basis should preserve the independent position-free map within 0.05 nat while reducing actual mapper bytes by at least 50%.

## T — Trial

- Source: EleutherAI/pythia-70m-deduped (GPTNeoX, RoPE fraction .25, theta 10,000).
- Target: HuggingFaceTB/SmolLM2-135M (Llama RoPE, theta 100,000).
- Wikitext-2 exact offset-aligned prompts; per-layer/head affine K/V ridge, alpha .01; calibration limited to positions 0–63.
- Development: seed 829, 24 calibration and 16 evaluation prompts. Rank 32 was selected as the lowest-NLL tested rank, then frozen with fresh prompt identities in commit 881b8992.
- Fresh: validation calibration and test prompts, 12 calibration and 16 test prompts for each of seeds 983, 997, 1009. Three distinct seeds, 16 test prompts each.
- Controls: native target prefill, direct RoPE-space ridge, independent inverse-RoPE content-space ridge, rank-32 shared PCA map basis.
- CPU only. GPU unavailable.

## D — Decision

**FAIL.** Fresh position-free full-map NLL improved over direct RoPE-space ridge in all three seeds by 0.65, 0.35, and 1.29 nat, but remained +0.82, +2.97, and +0.87 nat above native target prefill. The rank-32 shared basis was +12.35, +16.44, and +13.61 nat above native, and +11.53, +13.47, and +12.74 nat worse than independent content-space maps. It failed the 0.05 nat compression-quality gate in all seeds and the <=0.50 nat absolute gate in all seeds.

The shared payload was 573,060 B vs 2,996,100 B independent (80.87% fewer bytes), so storage passed while quality failed. Rank-32 cache relative MSE was 726–936 versus 0.320–0.346 for the full maps.

The map-family representation is native PCA and supplies no Mirror-specific advantage. No adoption claim.

## Facts

| Seed | Native target NLL | Direct RoPE ridge Δ | Position-free full-map Δ | Rank-32 Δ | Full-map bytes | Rank-32 bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 983 | 3.0854 | +1.4741 | +0.8245 | +12.3549 | 2,996,100 | 573,060 |
| 997 | 1.9113 | +3.3209 | +2.9746 | +16.4400 | 2,996,100 | 573,060 |
| 1009 | 2.8498 | +2.1645 | +0.8716 | +13.6069 | 2,996,100 | 573,060 |

Mean source prefill was 40.2 ms/prompt; native target prefill 104.3 ms; full translation 13.3 ms; rank-32 translation 12.9 ms; target suffix decode about 55.5 ms. These CPU timings exclude the rank-32 map reconstruction proxy of 23,961,600 MACs and do not establish an end-to-end latency win. Applying one reconstructed map family costs about 748,800 MACs/token.

All 48 test prompts were 16–115 tokens. Prompts with tested prefix beyond position 64 numbered 2, 2, and 4 by seed. Their native/full/rank32 NLL means were: seed 983 0.353/1.359/15.774; 997 0.756/2.611/21.289; 1009 1.304/1.599/18.730.

## Interpretation

Inverse RoPE helped the full independent mapper relative to the direct-space ridge on this pair, but did not bring it close enough to target-native quality. Compressing 180 layer/head/role maps into rank 32 destroyed their functional distinctions, even though the mapper payload fell by about 81%.

## C — Strongest counter-hypothesis

The primary limitation is cross-model hidden-state mismatch and resulting target trajectory drift, not positional phase alone. The small calibration set and single pair cannot distinguish that from a poor fit or role alignment. Rank-32 PCA may also be too restrictive, but it was the preregistered best development rank.

## U — Unconfirmed

- Exact fresh replay and post-run verification were not completed: the execution environment became unavailable after the fresh metrics were produced.
- The long-position subset is tiny; no long-context generalization claim.
- One ordered pair, one dataset, one next-token probe; no generation-trajectory or GPU result.
- A native PCA shared-map control exactly implements the same representation family; Mirror-specific gain is not established.

## Verification note

Three RoPE algebra tests passed before fresh evaluation, and model-cache inverse/reapply checks matched within 3.6e-15 (GPTNeoX) and 8.9e-16 (Llama). The exact fresh metric replay and final serializer test could not be run after environment loss. Treat the fresh metrics above as recorded single-run observations, not replay-verified results.