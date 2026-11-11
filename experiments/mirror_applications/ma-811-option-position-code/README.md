# MA-811 — option identity × sequence position

## H / T / D / C / U

**H:** Factorized option and position Views preserve unseen option-position roles and heldout four-step sequences at lower full bytes than an independent pair bank, while improving on ordinary additive factors.

**T:** Synthetic 8D shared core, six options, four positions, rank-2 residual atoms; 19 train and 5 heldout option-position pairs; six fixed four-step test sequences; seed 81101; zero optimizer updates. Compared independent pair matrices, shared core, native additive factors, and Mirror Views.

**D: FAIL.** Mirror Views achieve zero heldout sequence nRMSE and use 1,981 B versus 6,395 B for the independent bank (69.0% fewer bytes). The native additive-factor control also achieves zero error, uses 1,961 B, and has the same active compute. The native payload is 20 B smaller. Exact native factor alias fails the Mirror-specific gate.

**C:** Standard additive option and position coefficient tables explain all heldout behavior with fewer bytes. Mirror field naming/layout increases the serialized archive by 20 B.

**U:** Learned option policies, natural sequence distributions, real robots and physical latency. Fresh remained sealed.

## Evidence separation

- **Fact:** Full payload byte sizes, hashes, heldout sequence error, MAC proxy, and reconstruction checks are in `RESULTS_CORE.csv`.
- **Interpretation:** The synthetic pair bank is compressible through factorization, but the Mirror encoding is dominated by its ordinary additive control.
- **Hypothesis:** A useful position-aware option code must add a nonseparable role effect that remains cheaper than native per-pair or low-rank parameterization.
