# MA-599 status

- Status: **FAIL (development, verified)**
- Branch: `research/ma-599-factorized-hash-mirror-code-20261009`
- Development seeds 59901/59902 complete; fresh seeds 59911–59913 sealed.
- Implementation amendment 1 corrected a tensor conversion before any metrics; protocol gates unchanged.
- Tests: 3 passed. Deterministic replay pending.

## H / T / D / C / U

- **H:** A shared rank-4 bucket basis plus small expert codes would improve on tied/salted hashes by at least 1pp at no more than 1.05x salted actual bytes, and beat a byte-near ordinary rank-4 residual.
- **T:** Digits mod-4 conditional experts, 64→128→10 MLP plus shared router, seven methods, 800 AdamW updates; dev splits 59901/59902. Serialized NPZ payloads loaded for evaluation. Fresh seeds were not opened.
- **D:** **FAIL.** View is +.22pp / 0pp vs salted, not +1pp, and costs 27,651 B vs 10,735 B (2.576x). Native rank-4 residual is same accuracy at 17,337 B on seed 59901 and within .22pp on seed 59902. All four routes are used; every View oracle expert has >=.985 accuracy.
- **C:** Native hash salts and standard low-rank residuals explain the observed performance with far fewer bytes.
- **U:** Large MoE scaling, broader tasks, near-convergence capacity, efficient kernels, and language-model quality remain untested. The study uses fixed-update digits specialist classification.

## Facts / interpretation / hypothesis

- **Fact:** See `runs/dev_*/metrics.json` and `RESULTS_CORE.csv`; the factorized-view payload is 2.576x salted bytes in both dev splits.
- **Interpretation:** Useful logical specialists can share buckets, but this factorized View does not improve the quality/byte frontier over native addressing or ordinary low-rank corrections.
- **Hypothesis:** A bucket-space basis is too expensive at this small table size; sparse exception repair (MA-602) may better localize useful corrections.
