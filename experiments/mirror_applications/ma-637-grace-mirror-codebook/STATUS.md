# MA-637 status

- Status: FAIL for Mirror-specific attribution and all-snapshot storage gate
- Branch: `research/ma-637-grace-mirror-codebook-20261103`
- Base commit: `3fb5c60e`
- Protocol frozen before development: yes
- Development complete: yes (63701, 63702)
- Fresh/audit opened: no; native PCA aliases and early snapshots miss storage gate
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** A shared rank-4 value basis reduces growing GRACE codebook bytes without harming retention or scope; Mirror must beat native PCA coding after decode cost.

**T:** 128 sequential synthetic edits, 8D keys and 16D values in a true rank-4 subspace. Basis fitted on 16 separate calibration values. Snapshots at 16/32/64/128 edits; same nearest-key router/radius; explicit values vs basis/codes vs exact native PCA. Two development seeds. Measured full payload bytes, retention, positive/negative scope and compute proxy.

**D — FAIL.** Retention nRMSE was 0 for explicit memory and <1.1e-15 for coded memory; positive hit rate was 100%, false positives 0. Mirror bytes were 3,198–3,201 / 3,979–3,981 / 5,468–5,472 / 8,448–8,455 B at 16/32/64/128 edits. Explicit values were 3,456–3,457 / 4,912–4,919 / 7,823–7,833 / 13,657–13,679 B. The frozen ≤75% condition failed at 16 and 32, passed at 64 and 128. Code decoding increased operation proxy from 1,056,768 to 1,081,344 at 128 edits. Native PCA exactly matched arrays, bytes, retention and scope; fresh stayed sealed.

**C — Strongest counter-hypothesis.** The value saving is standard PCA compression of GRACE's explicit payload. Key lookup, radii and shared executor dominate small banks; coding becomes more useful as value state grows, but adds reconstruction work.

**U — Unconfirmed.** No real lifelong model editing, semantic keys, interference, edit aging, paraphrase retention or non-rank-4 value distribution was tested.

## Fact / Interpretation / Hypothesis

**Fact:** coded memory preserves synthetic retention and scope. Complete payload savings grow from ~7% at 16 edits to ~38% at 128. Native PCA is an exact alias.

**Interpretation:** low-rank value coding helps only after the memory bank grows enough to amortize its basis; it remains ordinary PCA coding and has added decode work.

**Hypothesis:** real GRACE edits may show different code rank and router cost; test only after a Mirror-specific code geometry is proposed.
