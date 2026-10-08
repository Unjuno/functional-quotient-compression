# MA-319 — Tucker matrix-bank Mirror layer coefficients

Status: **FAIL** under the frozen fresh storage and quality gates. Dedicated branch: `research/ma-319-tucker-matrix-bank-mirror-20261008`.

## H — hypothesis

Across four Transformer blocks, replacing each layer's free rank-2 Tucker coefficients with a shared-center/shared-radius circular Mirror view plus one angle per layer would reduce the complete serialized payload by at least 5% while keeping fresh character NLL within 0.02 nat/token of free rank-2 Tucker coefficients.

## T — what was run

**Fact:** Four-layer nanoGPT character models (4 heads, width 64, context 64) were trained for 500 AdamW updates at batch 32 and learning rate 0.001, with one CPU thread. Development seeds were 31901/31902. Valid fresh seeds were 31911/31912/31913. Tiny Shakespeare used its first 80% for training and middle 10% for development. Fresh NLL used Project Gutenberg *Pride and Prejudice* after normalizing fixed quotes/dashes and retaining only characters in the training vocabulary: 763,082 raw characters, 742,112 retained, 2.75% filtered; SHA-256 `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`.

Controls were full FP32 weights; free-coefficient Tucker banks at ranks 1, 2 and 4; and rank-2 circular Mirror coordinates. Four repeated block matrix families were compressed; all other model state, means, bases, codes, metadata and serialization overhead were paid. The full package was serialized with `torch.save`. See `PROTOCOL.json` and amendment A1.

**Protocol correction:** An implementation bug initially evaluated Tiny Shakespeare's final 10% during the first two development runs. Those logs are preserved but quarantined in `protocol_variants/contaminated_pre_amendment/`. The source was corrected and amendment A1 committed before valid fresh access. The corrected development path was tested not to read the audit file. No contaminated metrics are included below.

## D — decision and facts

**FAIL.** On all three fresh seeds Mirror used 358,587B versus 356,435B for rank-2 Tucker (+0.60%, 2,152B more), missing the required <=0.95x payload gate. Fresh NLL delta (Mirror minus Tucker rank-2) was +0.0123, +0.1164 and +0.0567 nat/token; the <=0.02 quality limit was missed in 2/3 seeds.

| Method | Mean package bytes | Fresh NLL, mean |
|---|---:|---:|
| Full independent FP32 | 849,287 | 2.4595 |
| Tucker rank 1 | 258,131 | 2.9675 |
| Tucker rank 2 | 356,435 | 2.6760 |
| Tucker rank 4 | 553,043 | 2.4595 |
| Mirror circle, rank 2 | 358,587 | 2.7379 |

Rank-4 reconstructs the four layer matrices nearly exactly and matches the full model's NLL within FP16 serialization. Rank-1 is smaller but substantially worsens NLL. Mirror's fit-operation proxy equals Tucker rank-2's 786,432 per checkpoint; this proxy covers low-rank projection/reconstruction arithmetic and excludes the shared SVD and small circle-coordinate extraction. CPU timing is diagnostic: Mirror throughput averaged about 0.93x rank-2 Tucker during audit evaluation, and materialization took under 1ms for both. Steady-state inference uses the same materialized nanoGPT layers.

## Interpretation and counter-hypothesis

**Interpretation:** Tucker sharing compresses repeated layer matrices substantially versus independent FP32 matrices. The circular code does not improve that frontier: two center values plus radius and angles replace free per-layer coefficients, but the extra center/radius arrays and archive keys cost more than the saved coefficient. It also incurs variable quality loss.

**C — strongest counter-hypothesis:** PA35's free Tucker coefficients already provide a compact direct address. The common-circle constraint is too restrictive for learned layer trajectories, while its metadata erases its theoretical per-layer code saving at four layers.

## U — unconfirmed

Deeper models, larger matrix banks, learned Mirror coordinates, other tensor decompositions, longer training, modern subword corpora, and runtime memory or latency benefits remain untested. This is one small character-LM compression screen, not a capacity claim.

## Verification

Five tests passed. The verifier reloaded all 25 serialized packages, checked exact payload hashes and reproduced every recorded NLL with maximum difference 0. Raw package files are local ignored artifacts; hashes and byte counts are retained in `RESULTS_CORE.csv` and per-seed JSON summaries.
