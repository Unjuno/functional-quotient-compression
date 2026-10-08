# MA-466 — UniPELT components as factorized Mirror axes

Status: **FAIL** (development screen; fresh seeds sealed)  
Evidence lane: MECHANISM/STORAGE/QUALITY/COMPUTE  
Branch: `research/ma-466-unipelt-factorized-mirror-axes-20261008`  
Base commit: `7cf891f`  
Prior art: PA84 (UniPELT)

## H — Hypothesis

Rank-two gates factored across task, layer and PEFT component could represent 128 logical adaptations with low output error and at least 25% fewer actual bytes than a full UniPELT gate table. The LoRA, bottleneck adapter and prefix ablations were required to check that all three axes contribute.

## T — What ran

The synthetic 4D model has 16 task IDs × 8 layers = 128 contexts. Three fixed/shared functions are available at every layer: a rank-one LoRA-like linear update, a two-wide tanh bottleneck adapter, and a prefix-like bias. A seeded rank-two CP tensor generates teacher gate logits. UniPELT learns a full 16×8×3 gate table; Mirror learns task, layer and component factors of rank two; the direct native CP control uses exactly the same map. Each model ran 2,500 Adam updates with 32 examples per update, then 64 heldout inputs per context. Development seeds: 46601 and 46602. Fresh seeds 46611–46613 remained sealed.

The payload includes all base matrices, LoRA factors, adapter weights, prefix vectors, gates or CP factors, and metadata. Query wall covers all 8,192 query vectors on CPU; it is a small synthetic batch and not stable deployment latency.

## Results — facts

| Seed | Method | Mean query RMSE | Actual bytes | MAC proxy/input | Mean active components | Train wall (s) |
|---:|---|---:|---:|---:|---:|---:|
| 46601 | UniPELT full gates | 0.02261 | 5,170 | 47 | 3.00 | 1.76 |
| 46601 | Mirror CP gates | 0.02122 | 4,356 | 56 | 2.99 | 2.55 |
| 46601 | native CP gates | 0.02122 | 4,356 | 56 | 2.99 | 2.42 |
| 46601 | shared global gates | 0.07084 | 3,645 | 47 | 3.00 | 1.62 |
| 46602 | UniPELT full gates | 0.02541 | 5,170 | 47 | 3.00 | 1.68 |
| 46602 | Mirror CP gates | 0.02138 | 4,356 | 56 | 2.99 | 2.49 |
| 46602 | native CP gates | 0.02138 | 4,356 | 56 | 2.99 | 2.37 |
| 46602 | shared global gates | 0.07815 | 3,645 | 47 | 3.00 | 1.56 |

Mirror uses 15.8% fewer bytes than full UniPELT (4,356 vs 5,170), short of the frozen 25% requirement. Its gate-generation MAC proxy is 19.1% higher (56 vs 47), within the 25% compute bound. Mean output RMSE was slightly lower in both seeds. All three components were useful: removing them raised RMSE to LoRA 0.2040/0.2833, adapter 0.1276/0.1376, and prefix 0.2085/0.1681 in seeds 46601/46602, respectively. Nearly all components remained active; this is not sparse execution.

The direct native CP model had identical serialized bytes, hash and context outputs to Mirror on both seeds. Thus any quality or storage effect is ordinary CP gate factorization, not a Mirror-specific freedom. The shared-gate baseline used fewer bytes but had much worse quality. Full per-context metrics, component ablations, actual hashes, compute and timing are in `RESULTS_CORE.csv` and `runs/dev_*`; `verification_report.json` records exact replay.

## D — Decision

**FAIL.** The frozen storage gate is missed, and the native CP control exactly aliases Mirror. The mechanism did preserve quality while reducing gate-state size, and the ablations confirm that all three PEFT components matter in the synthetic function bank. But the complete payload saving is modest because shared component tensors dominate, the extra gate calculation raises operations, and no Mirror-specific gain is established. Fresh seeds stayed sealed.

## C — Strongest counter-hypothesis

This is low-rank CP factorization of UniPELT's gate tensor. The exact native control reproduces all predictions and bytes. The small quality improvement over a full gate table at fixed updates can be a regularization/optimization effect, not extra capacity.

## U — Unconfirmed

This is not a UniPELT Transformer or language-model evaluation. It does not establish near-convergence capacity, real attention/prefix behavior, deployment latency or a wider storage frontier. Fresh seeds were not opened.
