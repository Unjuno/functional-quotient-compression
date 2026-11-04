# MA-639 status

- Status: FAIL for frozen payload gate and Mirror-specific attribution; useful separation of locality/compression
- Branch: `research/ma-639-loki-nullspace-mirror-codes-20261103`
- Base commit: `a7d4bff6`
- Protocol frozen before development: yes
- Development complete: yes (63901, 63902)
- Fresh/audit opened: no; code payload exceeded 75% gate and native PCA aliased
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** LOKI's null-space projection reduces protected-key interference independently of Mirror code compression; a shared value basis then compresses updates without damaging locality, and must beat native PCA.

**T:** Synthetic 32×32 layer, 16 edits and 16 protected keys. Evaluated aligned rank-2 and private rank-16 value worlds; basis fit on 12 edits; swept stored ranks 0/1/2/4/8/16. Compared unprojected full edits, LOKI-projected full edits, projected Mirror values and native PCA. Actual full payload bytes, held-out efficacy and protected-key leakage measured on two development seeds.

**D — FAIL.** LOKI projection reduced protected-key leakage from 0.40 for unprojected updates to 0 while preserving exact edit efficacy. In the aligned rank-2 world, rank-2 Mirror values had zero efficacy error and zero leakage, at 8,012–8,029 B versus 9,368–9,382 B projected full updates (about 14.5% savings), missing the frozen ≤75% payload gate. Native PCA exactly matched Mirror arrays, bytes, efficacy and leakage. In the private rank-16 world, even stored rank 16 had held-out nRMSE 0.502–0.576 because 12 training edits did not identify the full private basis; projected full updates remained exact. Fresh stayed sealed.

**C — Strongest counter-hypothesis.** LOKI projection explains locality; ordinary PCA explains the small value compression. The rank-16 miss reflects insufficient basis-training edits rather than a Mirror-specific capacity limit.

**U — Unconfirmed.** No pretrained model, real LOKI optimizer, semantic facts or non-orthogonal protected keys were tested.

## Fact / Interpretation / Hypothesis

**Fact:** null-space projection removed protected-key leakage independently of value compression. Aligned low-rank codes reduced full payload by ~14.5%; private rank 16 was not recovered from 12 training edits. Native PCA was an exact alias.

**Interpretation:** storage and interference are separate axes. The proposed Mirror code added no distinction from PCA and missed the frozen full-payload gate.

**Hypothesis:** learned null-space geometry on real update gradients might create a more compressible value family, but requires a new protocol and native low-rank control.
