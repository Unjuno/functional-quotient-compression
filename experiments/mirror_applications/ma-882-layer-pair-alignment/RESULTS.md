# MA-882 Results

## H — Hypothesis

With target injection layers and per-layer top-2 routes held fixed, factorizing affine KV translator maps over model-pair identity, source-depth anchor, target layer, and KV role will preserve heldout NLL relative to the full selected-map bank and ordinary PCA while reducing complete inference payload bytes.

## T — Test

Two ordered pairs shared SmolLM2-135M as target: Pythia-70M→SmolLM2 and Pythia-160M→SmolLM2. Three normalized source-layer anchors (0, 0.5, 1) were considered for each of 30 target injection layers. Disjoint calibration prompts selected two source anchors per target layer using mean normalized K/V reconstruction error; the chosen routes and equal weights were fixed across full, PCA, and CP map representations. All controls used ordinary unweighted affine ridge (alpha 0.01) and inverse/source plus target RoPE handling. This is not an attention-weighted CacheBridge reproduction.

Development used 24 map-fit, 8 route-selection, and 16 test prompts per pair on Wikitext-2 train. No PCA rank met the development quality rules, so rank 256 was frozen as the lowest-NLL fallback. No CP rank met the quality rules either; CP rank 8 was frozen as its lowest-NLL fallback. Fresh calibration used 12 map-fit and 4 route-selection prompts per pair and seed. Fresh tests used 12 prompts per pair across seeds 983, 997, and 1009. All target layers were injected in every representation. CPU only; CUDA unavailable.

## D — Decision: FAIL

**Facts.** The frozen CP view missed the preregistered quality gate in all 6 pair×seed conditions. Its NLL was +13.06 to +19.76 nat versus full matched-route maps, +8.78 to +15.53 nat versus PCA, and +14.46 to +19.82 nat versus native target prefill. Mean NLL across the three fresh seeds was 20.033 nat for CP, 8.253 for PCA, 3.988 for full top-2 maps, 4.561 for fixed nearest-depth maps, and 2.660 for native. The full top-2 map control itself exceeded native by more than 1 nat in 4/6 pair×seed conditions.

The complete CP payload was 4,286,304 B versus 11,981,134 B full (64.2% fewer) and 5,014,068 B PCA (14.5% fewer). All 6 storage checks passed; all 6 quality checks failed. The full top-2 layer matcher beat fixed nearest-relative-depth mapping in all 6 pair×seed conditions, by 0.02–0.99 nat. This supports the value of pair-calibrated layer selection in this screen, while the compressed views failed to preserve it.

Serialized payloads were decoded and replayed on the same fresh test prompts under a protocol amendment. Payload bytes and hashes matched exactly; decoded outputs reproduced each method's pair-mean NLL within 0.00072 nat. Native cache replay identity differed from native prefill by at most 1.46e-5 nat. Average one-time payload decode was 7–12 ms for full, 15–47 ms for PCA, and 21–59 ms for CP. Mean per-prompt map application was about 0.05 s, and target replay about 0.05 s. Map fitting plus route selection took 11.1–12.6 s per fresh seed; CP fitting took about 2.0–2.2 s for 300 updates. Development took 293.5 s; peak RSS was 2.7 GiB. Overall fresh-run wall-clock was not recorded by the initial runner. CUDA unavailable.

**Interpretation.** The top-2 source-layer match improves on fixed relative-depth selection, but the learned map family is fragile under compression: even PCA rank 256 loses about 4.3 nat against full maps, and the selected CP code loses another 11.8 nat on average against PCA. CP meets the storage objective but lands far outside the quality frontier. It is an ordinary tensor factorization and gives no Mirror-specific advantage.

## C — Strongest counter-hypothesis

The selected CP rank and optimization budget may underfit a highly sparse, nonconvex coefficient field. Development itself found no quality-qualified PCA or CP representation, and CP rank 32 was worse than rank 8, consistent with unstable optimization. Also, full cross-model maps are not reliably useful versus native prefill: their native delta exceeds +1 nat in 4/6 conditions. These limits mean the experiment does not isolate CP factorization as the sole source of quality loss.

## U — Unconfirmed

Only two pairs sharing a target, three fixed source-depth anchors, one-token replay, and Wikitext-2 were tested. There is no unseen-pair result, no attention-weighted ridge control, no larger CP-rank or optimizer sweep, no private residual/fallback, no multi-token trajectory test, and no GPU measurement. Pair-specific Wikitext rows overlap across the two pairs in some calibration and test chunks, so pair conditions are correlated; seed chunks remain disjoint within each pair.

## Payload replay amendment

The first fresh run evaluated in-memory decoded maps but only measured serialized bytes. After it completed, the protocol was amended to add deterministic full/PCA/CP payload decoders and replay the same frozen audit prompt IDs from the saved payloads. Ranks, routes, prompts, and hyperparameters did not change. This replay verifies serialization and metric reconstruction but is not an independent audit. The pre-amendment results remain in `fresh_results_pre_payload_roundtrip.json`.

## Provenance

Protocol and implementation were frozen at `ca28acaf8`; development ranks were frozen at `4350280f0`. `fresh_results.json` and `payload_roundtrip_replay.json` contain prompt IDs, per-pair metrics, byte hashes, timings, and replay differences. See `source/verify_results.py` for integrity and gate replay.
