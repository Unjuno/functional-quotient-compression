# MA-247 long-training reevaluation (exploratory; 2026-10-16)

MA-247 originally failed its frozen 1,200-update development gate. This new, separate CPU experiment tested whether more shared-weight training changes that outcome. It does **not** modify the old FAIL, the official MA registry, or main.

## Protocol

Three newly generated synthetic aligned teacher worlds (24931, 24932, 24933), two initializations per world, five methods, 8,192 unique training inputs, 2,048 validation inputs, 4,096 test inputs. Each model uses 9,600 AdamW updates (learning rate 0.01), minibatch size 64. Measurements were fixed at updates 300, 1,200, 2,400, 4,800 and 9,600. The methods share the same random initial shared weights and the same sampled minibatch-index sequence. Historical MA-247 reserved fresh worlds 24701-24703 were not used.

| Method | Trainable params | Saved inference bytes | Test MSE, 1,200 updates | Test MSE, 9,600 updates |
|---|---:|---:|---:|---:|
| Recurrent Mirror | 264 | 3,297 | 0.001480 | 0.00006389 |
| Same-count scalar gate | 264 | 3,366 | 0.00064789 | 0.00021330 |
| Static rank-1 LoRA | 356 | 3,938 | 0.00139354 | 0.00017015 |
| Hard tied | 260 | 3,043 | 0.00146014 | 0.00023742 |
| Four untied blocks | 1,028 | 6,181 | 0.00009287 | 0.00009106 |

MSE entries are the median of the three world means (two initializations per world). At 9,600 updates Mirror beat scalar gating and LoRA in all 3 worlds and all 6 paired seeds. This is a limited synthetic result, not a general compression or language-model claim. The teacher itself was constructed using the same Givens Mirror family, favoring Mirror structurally.

## Compute, data and batch follow-up

On AMD EPYC 9V74, PyTorch 2.10 CPU FP32, one thread per worker with three concurrent workers, median [range] training wall seconds for 9,600 steps: Mirror 20.52 [19.49,21.46], scalar gating 9.35 [8.77,9.52], LoRA 10.65 [10.02,10.80]. Clock was not locked. Thus the Mirror quality improvement is **not** an equal-wall-time victory.

A separate ablation used two new worlds (24941,24942), two initializations, three methods, and four configurations (48 fits): 8k vs 32k unique examples with batch64/2,400 steps; batch256/600 steps with 153,600 presentations matched to batch64/2,400; and batch256/2,400 steps (four times the presentations). More training data alone did not reliably help Mirror; increasing batch size while reducing optimizer steps at fixed example presentations harmed its test loss. These are preliminary findings with only two worlds and a common learning rate.

## Evidence and scope

Seven unit tests passed. Thirty primary and 48 ablation models were checked for finiteness, duplicate records, and frozen-source hashes. A complete re-training of one Mirror and one LoRA model matched all five recorded evaluation checkpoints and model tensor SHA256 values. Full reproducible code, fixed protocol JSONs, per-run histories, audit, raw data, and the Japanese report are provided as the conversation artifact `MA247_LONG_BUDGET_20261016_FULL_EVIDENCE.zip` (SHA-256: `67feb80aec721c22f8869b2e2a151c0660f4b7bf3f45386d9a3700877245bac8`). Code has not been published in this repository snapshot; do not treat this note alone as a complete reproducible GitHub release.

**Research conclusion:** The original MA-247 FAIL applies to the frozen short-run protocol only. In the new aligned synthetic worlds, longer shared training reverses the ranking against LoRA and scalar gating. It does not identify gradient interference as the cause; a follow-up requires per-loop gradient contributions and equal-time/equal-byte/equal-compute frontiers.
