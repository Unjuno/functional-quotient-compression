# MA-622 status

- Status: FAIL (development screen; Mirror-specific claim rejected)
- Branch: `research/ma-622-cavia-factorized-context-20261103`
- Base commit: `739ba2ed`
- Protocol frozen before development: yes (`freeze.json`)
- Development complete: yes (seeds 62201, 62202)
- Fresh/audit opened: no; development specificity gate failed
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H — Hypothesis.** A factor-addressed shared CAVIA context can recover held-out factor combinations with lower per-task state than independent contexts; a Mirror-specific claim requires beating native direct-angle and additive CAVIA controls.

**T — Trial.** Synthetic 16-input, 8-context linear mechanism; 4×4 task-factor pairs, diagonal held out; 16 support and 128 audit examples per task; aligned and private-interaction worlds; development seeds 62201/62202. Compared zero context, support-fitted independent CAVIA, factorized rotation view, identical native direct-angle control, and additive factor tables. Actual `.npz` payload bytes measured. No optimizer updates.

**D — FAIL.** In aligned world, factorized view achieved held-out nRMSE 0.000 on both seeds; state was 320 B versus 1,024 B for independent contexts. Complete payload was 2,272–2,275 B versus 2,789–2,791 B. However, native direct-angle control matched output and state exactly, so the Mirror-specific criterion failed. Additive factorization used fewer task-state bytes (512 B) but had nRMSE 0.156–0.258. Fresh seeds stayed sealed under the frozen gate.

**C — Strongest counter-hypothesis.** This is ordinary structured CAVIA context parameterization: identical direct-angle coordinates reproduce the Mirror interface exactly. Observed aligned gains support factorized context compression for this synthetic generator, not a distinct Mirror advantage.

**U — Unconfirmed.** No fresh-seed replication, meta-training, noisy support, natural task families, or language-model quality. Private interaction showed nRMSE 0.141–0.198, demonstrating a gap relative to independent adaptation but not quantifying the minimum private residual needed.

## Evidence separation

**Fact:** aligned held-out nRMSE was 0 for factor rotation and native direct-angle methods; serialized payload and task-state bytes matched exactly.

**Interpretation:** structured factor contexts compress this aligned synthetic family versus independent task contexts, while additive tables trade quality for smaller state. The native angle alias removes Mirror-specific attribution.

**Hypothesis:** similar factor structure may help real meta-learning tasks if the factors are known or inferable and the rotation code generalizes. This remains untested.
