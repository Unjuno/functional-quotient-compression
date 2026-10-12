# CMW-001 — Controlled causal-event world dynamics with shared Mirror views (2026-10-12)

**Verdict: DEVELOPMENT GATE FAIL, fresh 18021–18025 unopened.** This is a standalone research intake linked to MA-798/799/802/804, not a newly completed registered MA candidate. `main` and canonical worker registry unchanged.

**Mirror insertion:** A shared pair-relation-to-impulse feed-forward network takes a compact action-type View code, mediated by three shared learnable 2D rotation planes before its GELU activation. The code changes the logical transition function without storing four entire transition FFNs. Output is an impulse to target and opposing recoil to actor.

## Scientific distinction
The experiment trains forward transition models with supervised **actor, target, action-kind labels** (known inputs to the forward model). The reverse event is **not given at inference**: score all 48 candidate actor×target×kind transitions against a pair of observed before/after full states and choose minimum residual. This is event *inference* via supervised learned dynamics, not unsupervised discovery of latent actions and not prediction of an unknown future random action. Correlation is not enough for identification in the real world; intervention contrasts below are controlled synthetic do-actions.

## Data and train protocol
- 4 synthetic mass-bearing objects, each position(2), velocity(2), mass(1): full observation width20; 12 directed actor-target pairs and four kinds (repel, attract, clockwise, counterclockwise): 48 distinct labelled event classes.
- Nine complete actor-target-kind combinations held out from fitting and validation; the other 39 combinations train and select checkpoints. Each world: train4000, val800 (seen classes), test seen1000, heldout1000. Heldout observation pairs used only for final metrics, never selecting hyperparameters or checkpoints.
- Worlds: `shared` (common relative motion law); `partial_shift` (specific actor/type nonlinear exceptional force); `unrelated` (unobserved triple-specific independent force offsets). Unknown private offsets cannot in general be recovered from unseen signatures.
- Nine methods: native conditioned MLP, FiLM, native shared low rank, fixed disjoint planes, identical-initialized random frozen planes, learned shared planes, native independent action-kind heads, native geometric relative-direction 2×2 operator, and selective Mirror/direct-gate mixture. All use identical observed relative state features and actor/target scatter physics.
- Calibration 9×4=36 models ×300 updates using seed17000 and **seen-class** validation only; chosen per-method hyperparameters locked before development. Development 3 seeds (17011..17013) ×3 regimes ×9 methods =81 models ×650 updates. **117 training jobs / 63,450 optimizer updates.** AdamW, FP32 CPU, one Torch thread. Whole-bank saved compressed NPZ includes every learned parameter, action code, plane vectors, kind-specific state, and metadata.

## Development results (3-seed means)

| World | Method | Held-out impulse nMSE ↓ | Complete event top-1 ↑ | Full inference NPZ |
|---|---|---:|---:|---:|
| shared | frozen random shared planes | 0.01597 | 99.67% | 7,468 B |
| shared | **learned shared Mirror planes** | **0.01206** | **100.00%** | 7,467 B |
| shared | native kind-specific readouts | 0.00725 | 100.00% | 7,490 B |
| shared | **native geometric relative-vector operator** | **0.00459** | **100.00%** | **6,510 B** |
| partial_shift | learned Mirror planes | 0.04257 | 97.33% | 7,472 B |
| partial_shift | native geometric operator | 0.03020 | 97.56% | 6,511 B |
| unrelated | learned Mirror planes | 0.53832 | 39.33% | 7,463 B |
| unrelated | native geometric operator | 0.48962 | 45.00% | 6,509 B |

In the shared-law world, Mirror improved heldout nMSE vs exactly matched frozen-random planes in all three seeds (absolute gains 0.005005 / 0.004986 / 0.001727), but native geometric control was better in **both** quality and bytes. The oracle with true simulator parameters recovers all hidden single events in all 9 world/seed combinations; poor model inference in off-family worlds is a generalization limit, not strict intrinsic observational ambiguity in this simulator.

**Controlled do-intervention contrast nMSE:** learned Mirror 0.01008 vs native geometry 0.00341. **Novel pair of simultaneous events nMSE:** 0.01186 vs 0.00335. **Conditioned 6-step rollout RMSE:** 0.00989 vs 0.00549. The two-event ground truth composes **additively**; this is not nonadditive causal composition. True hidden confounders were not introduced.

## Original preregistered gate, NOT changed after testing
Shared-world learned-vs-frozen absolute heldout nMSE improvement >=0.05 in at least 2/3 development seeds, within 0.01 nMSE of native and hidden event top-1 >=50%. First criterion passed 0/3, so **FAIL_DEV_GATE**. All original fresh seeds 18021–18025 remain unopened; no new promotion or MA status change.

## Runtime and full representation costs
AMD EPYC 9V74, Torch 2.10.0+cpu, one thread FP32, 20 warmups+70 repetitions, batch1 & batch64 measured standalone after training. Shared-world batch1 eager median learned Mirror **0.381 ms** vs native geometry **0.218 ms**; batch64 **0.438 ms** vs **0.252 ms**. No runtime gain. Simple selective Mirror/direct gate used mean Mirror proportion around 0.38 but was less accurate and used ~8,187B; this gate used **action-type only**, not dynamic anomaly detection.

## Integrity and actual evidence boundary
16/16 unit tests passed; all **81 actual saved inference NPZ**, bytes and SHA256 verified; **891 individual saved-checkpoint scalar metrics** recomputed max discrepancy 1.11e-16; one entire learned-Mirror training rerun reproduced **the identical NPZ SHA256**. Frozen source+protocol SHA256: `46dec0ae0d63ea2c831277939e4aab1465282251536063a38bba28f43a0ca72a`. Full standalone 81-model artifact is available in the conversation as `CMW001_2026-10-12_FULL_EVIDENCE.zip` SHA256 `d5ad84098d64277d4daa43fd73aa4659fd4f59da8addfd9477448847228400b0`. Source/results-only artifact: `CMW001_2026-10-12_SOURCE_RESULTS.zip` SHA256 `b94d191a2b7a05f62d5d2ae82018c78c74bd4875950807696d2c0e13ee1dbcdb`.

These are **synthetic known-coordinate observations**, not real-world sequential video, language, naturally emerging actors, unlabeled latent action discovery, or universal causal identification. The experiments use no native full Mamba or pretrained world-model comparators. Static random frozen planes are not a native symmetry-breaking guarantee; learned-axis advantage is only over that matched parameterization. The native geometric baseline is a stronger mechanistic prior.

## Next falsifiable direction
The positive native geometric result suggests inserting `m` **in the relation/effect transition space** (learned force directions) rather than as an arbitrary 24D FFN rotation. A state-conditioned gate must determine when shared transformation is valid and allocate optional private residuals for exceptions; the current action-type-only gate does not do this. Freeze a new pilot protocol with dedicated dev/fresh partitions, compare against simple native 2×2 conditional operators, ordinary FiLM/LoRA and state-conditioned residuals, and include missing-actors/hidden confounders and *unlabeled event inference* as independent tasks. Do not retune current opened dev/fresh.
