# MA-845 status

Status: **FAIL** for the preregistered byte/fallback advantage; scoped synthetic sequential-memory screen. Fresh remained sealed after exact native fallback alias and strict byte-gate miss.

## H / T / D / C / U

- **H:** Rank-4 codes for in-span memories plus private fallbacks for exceptions would preserve retrieval while reducing full bytes by >=25% and beating SERAC/GRACE and native code growth.
- **T:** One synthetic seed; 24 sequential 32D memories; basis fitted from the first eight, 16 in-span then eight exceptions; fixed .02 residual threshold; zero optimizer updates. Focused test: 1 passed.
- **D:** FAIL. The hybrid stores 16 codes and 8 private vectors (fallback rate 8/24=33.3%), reaching retrieval nRMSE .00471, max per-memory .01455 and specificity1. Its complete payload is 3,615 B, only 2.1% below independent traces (3,694 B) and 11.3% below SERAC/GRACE-style explicit memory (4,075 B). Shared-only state is 1,917 B but nRMSE .5645 and max error ~1.0. Mirror and native low-rank/private-fallback payload hashes and metrics match exactly. Fresh remained sealed.
- **C:** The private exceptions consume one third of entries, and at this bank size basis, code, flag, ID and Fisher metadata overhead erase most savings. The fallback policy is ordinary low-rank projection with a private-vector threshold.
- **U:** No semantic model, actual SERAC/GRACE system, natural memory arrival distribution, larger bank scaling, fresh seeds or production latency. Wall time is only a local array-copy timing, not a serving benchmark.

## Fact / Interpretation / Hypothesis

- **Fact:** Hybrid memory passes this synthetic retrieval threshold and uses 3,615 B; independent uses 3,694 B. The private fallback count is eight. Native low-rank fallback exactly matches Mirror.
- **Interpretation:** Shared codes preserve this constructed in-span cohort, but code metadata plus private exceptions yield only 2.1% byte savings at 24 memories. The no-fallback frontier is much smaller but loses retention.
- **Hypothesis:** Larger repeated in-span memory banks may amortize the shared basis, while higher exception rates will reduce the benefit; this screen does not establish either trend.
