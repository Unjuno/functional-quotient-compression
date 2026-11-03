# MA-636 status

- Status: FAIL for Mirror-specific attribution and frozen storage gate
- Branch: `research/ma-636-serac-mirror-value-code-20261103`
- Base commit: `5f080ba7`
- Protocol frozen before development: yes
- Development complete: yes (63601, 63602)
- Fresh/audit opened: no; native PCA alias and 50% storage gate miss
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** A shared rank-4 value basis reduces SERAC counterfactual storage while preserving scope routing; Mirror must beat native PCA coding after accounting for retrieval and decode.

**T:** Synthetic SERAC memory with 128 keys, 16D values, 96 basis-training edits and 32 held-out edits; rank-4 value family; four positive and four negative queries per key. Compared explicit memory, Mirror PCA values and native PCA values with an identical nearest-key/radius router. Two development seeds; full payload bytes, scope, efficacy, operation proxy and measured retrieval/decode time recorded.

**D — FAIL.** Scope hit rate was 100%, negative false-positive rate 0, and held-out value nRMSE was 0 for explicit values and <5e-16 for coded values. Mirror payload was 8,071–8,080 B versus 13,293–13,316 B explicit (about 39% smaller), missing the frozen 50% threshold. Operation proxy rose from 1,056,768 to 1,081,344 because of decode. Native PCA used exactly the same arrays/bytes and reconstruction; its measured latency was lower in both seeds. Fresh stayed sealed.

**C — Strongest counter-hypothesis.** This is standard PCA/low-rank compression of SERAC counterfactual values. Shared basis coding did not halve complete memory bytes because keys, routing metadata and base executor remain paid.

**U — Unconfirmed.** This synthetic scope router does not test pretrained-model edit success, paraphrases, real scope boundaries, interference or retrieval indexes at production scale.

## Fact / Interpretation / Hypothesis

**Fact:** code memory preserved the tested scope and values while reducing complete bytes by about 39%; native PCA exactly matched and used lower measured latency.

**Interpretation:** value compression is useful but misses the frozen storage gate and gives no Mirror-specific gain; routing and shared executor are a large fraction of the total.

**Hypothesis:** real edit banks with higher redundancy or compact retrieval indexes may change the full-payload frontier; must compare against explicit SERAC memory and native codecs.
