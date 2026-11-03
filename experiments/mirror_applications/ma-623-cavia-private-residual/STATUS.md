# MA-623 status

- Status: FAIL (development screen)
- Branch: `research/ma-623-cavia-private-residual-20261103`
- Base commit: `3a9c7b05`
- Protocol frozen: yes (`freeze.json` SHA-256 matches)
- Development complete: yes (62301, 62302)
- Fresh/audit opened: no; joint gate failed
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H — Hypothesis.** Low-dimensional task-private residual coordinates on a shared factorized CAVIA context recover held-out interactions at lower actual serialized bytes than independent full contexts; residual rank locates the quality/storage break-even.

**T — Trial.** Synthetic 16-input, 8-context linear model, 4×4 factor pairs with diagonal held out, 16 support and 128 audit examples per task. Aligned and rank-2 private-interaction worlds, two development seeds. Swept residual ranks 0, 1, 2, 4, 8. Compared shared factor view, PCA residual, direct coordinates, and independently support-fitted full CAVIA. Complete compressed `.npz` payloads include predictor, context basis, factor codes, residual state and pair indices. Fresh seeds 62311–62313 were not accessed.

**D — FAIL.** In the aligned world, rank 0 reproduced held-out tasks at numerical-zero nRMSE. In the private-interaction world, rank 2 nRMSE was 0.099–0.109 at 3,347–3,351 B; rank 4 was 0.044–0.079 at 3,755–3,762 B; rank 8 reached zero error at 4,527–4,529 B. Independent CAVIA achieved zero error at 3,039–3,045 B. No rank on both seeds met both nRMSE ≤0.05 and ≤75% of independent payload. The Mirror PCA residual exactly matched the native PCA residual control in metrics and bytes.

**C — Strongest counter-hypothesis.** Ordinary PCA residual adaptation explains all gains. The interaction family is deliberately low-rank, yet native PCA residual ties the Mirror construction exactly. Independent support-adapted CAVIA also achieves exact held-out fit with a smaller full payload than the rank needed by shared-plus-private state.

**U — Unconfirmed.** This is a small noiseless linear mechanism screen. It does not test meta-training, noisy supports, natural task families, or language quality. Rank 4's seed variation and the minimum useful rank need fresh replication only in a separately justified follow-up.

## Fact / Interpretation / Hypothesis

**Fact:** aligned tasks required no private residual. In the private-interaction world, full-rank residuals were needed for exact recovery on both seeds, and their payload exceeded independent CAVIA. The rank-4 quality gate passed on only one seed. Native PCA residual was an exact method alias.

**Interpretation:** private state can restore held-out functions, but in this tested setting the state required for reliable quality removes the storage advantage. Low-rank residual adaptation is not Mirror-specific.

**Hypothesis:** structured natural task interactions could have lower residual rank and change the storage frontier. That remains untested.
