# MA-618 status

- Status: FAIL on development byte and compute gates; fresh sealed.
- H: add new functions to a frozen interpreter using signature+phase codes and sparse private residuals.
- T: 2 development seeds; 32 functions/world (24 on-orbit, 8 private); 16 support and 512 query points; no optimizer updates.
- D: quality passes (Mirror mean nMSE 1.33e-6–1.88e-6), but Mirror uses 2,777 B vs direct coefficients 2,841 B (only 2.3% saving) and full Fourier codes 2,597 B. Fit proxy 13.27M vs 49,152 direct; measured total fit+query 12.6–13.2 ms vs 6.9–7.4 ms direct and 2.4–2.9 ms full.
- C: fixed Fourier structure explains the signal; full coefficients serialize more efficiently and are more accurate.
- U: learned interpreter and natural tasks, fresh seeds, scaling, other bases, optimized kernels.
- Amendments 1 and 2 corrected support residual fitting and added timing; both were recorded before fresh access.
