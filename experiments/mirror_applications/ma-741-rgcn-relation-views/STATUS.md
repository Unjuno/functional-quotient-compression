# MA-741 status

- Status: FAIL (scoped synthetic Mirror-specific claim)
- Branch: `research/ma-741-rgcn-relation-views-20261109`
- Base commit: `bf06021e`
- Development mechanism screen: complete, seed 74101; 10 result rows.
- Fresh/audit: unopened after exact native rank-2 coefficient alias.
- Verification: three tests passed; deterministic replay recorded below.

## H / T / D / C / U

- **H:** Relation Views beat free R-GCN relation coefficients on storage and held-out links.
- **T:** 32 synthetic relations, dimension 16, four shared matrices, rank-2 code factorization, 24 fit/eight held-out; full matrices, free coefficients, native factorization, Mirror and universal controls.
- **D:** **FAIL.** Held-out ranking is perfect in the structured family, but Mirror saves only 0.5% versus free coefficients and exactly aliases native rank-2 factorization.
- **C:** Standard low-rank factorization already represents all Mirror functions measured.
- **U:** Real R-GCN training, noisy relation metadata, natural graph benchmarks and relation few-shot adaptation.
