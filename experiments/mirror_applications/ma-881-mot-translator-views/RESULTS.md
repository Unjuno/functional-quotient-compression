# MA-881 Results

## H — Hypothesis

Two token-routed cross-model KV translator experts, trained with target-side Context Correction, can be represented by one shared rank-32 map basis plus small per-expert coordinates while preserving heldout target quality and reducing serialized translator bytes by at least 50% in each fresh world.

## T — Test

The frozen component-level screen used Pythia-70M → SmolLM2-135M, affine weighted-ridge KV maps, a shared 16-unit token router with deterministic straight-through Top-1, 24 router updates, and target query KV replay correction. Development used Wikitext-2 train seed 829 and selected rank 32 by lowest shared-view NLL among 4/8/16/32. Fresh validation calibration and test prompts were disjoint across seeds 983, 997, and 1009; each had 4 prompts per expert, 4 router/correction prompts, and 12 test prompts. Controls were native target prefill, single full ridge, and independent full two-expert MoT-style translators with the same router/correction objective. CPU only; CUDA unavailable. This is not a full MoT reproduction.

## D — Decision: FAIL

**Facts.** Across all three fresh seeds, shared-view NLL was 18.14, 22.57, and 17.38 nat. Full two-expert MoT-style controls scored 6.60, 4.38, and 6.90 nat; single-ridge and native target scores were 4.07, 4.31, and 5.04 nat. Shared-view missed every quality limit: it was 10.48–18.19 nat worse than full MoT-style and 12.34–18.26 nat worse than single ridge/native. Context-correction relative MSE was 1.44, 1.50, and 1.70. The router used both experts in every seed (expert-0 fractions 0.748, 0.288, 0.869), so total expert collapse does not explain the failure. The full MoT-style control itself exceeded native NLL by 0.07–2.53 nat.

The shared payload was 600,656 B against 5,995,854 B for the full bank in all worlds, an 89.98% reduction. Payloads include the map basis and mean, both expert coefficient arrays, role references/metadata, router weights, pair ID and format header. Mean router training time was 6.64 s for shared maps and 10.82 s for the full bank; test router application and target replay timings are in `fresh_results.json`. Peak RSS was 1,549.5 MiB. Each world used 12 fresh target prompts. Integrity and preregistered gate checks pass; all three quality gates fail and all three storage gates pass.

**Interpretation.** The low-rank shared representation is highly storage-efficient but destroys useful translation behavior even at selected rank 32. Since the representation is ordinary PCA coefficient coding, no Mirror-specific benefit is established. The failing full-MoT control indicates that small disjoint expert calibration plus the router/correction procedure is itself a weak baseline on this pair; the screen therefore does not isolate low-rank compression as the only source of quality loss.

## C — Strongest counter-hypothesis

The result may be driven by an underfit MoT-style setup: four prompts per expert and four router/correction prompts are too few for 120 target KV map roles, while 24 updates and a hard straight-through gate may not optimize routing reliably. Supporting evidence is that even the independent full-bank control is worse than native in all three seeds, although the single full-ridge baseline is close to native.

## U — Unconfirmed

No full MoT translator backbone, recurrent multi-token cache trajectory, larger calibration budget, alternate router schedule, matched-byte native low-rank control, GPU latency, or second model pair was tested. The experiment does not determine whether a well-trained MoT bank can be compressed this way.

## Provenance and reproduction

Protocol/source were frozen at `55146aac7`; the development rank was frozen at `c0d540a58` before fresh evaluation. `fresh_results.json` stores prompt row IDs, model provenance, per-world timings and gate metrics. Serialized fresh payloads are under `artifacts/`. Run `python experiments/mirror_applications/ma-881-mot-translator-views/source/verify_results.py` and the local tests for integrity checks.
