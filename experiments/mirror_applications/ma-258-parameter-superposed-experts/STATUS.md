# MA-258 status

- Status: PROMISING, scoped to aligned single-plane Givens experts
- Branch: research/ma-258-parameter-superposed-experts-20261009
- Fresh: 3 worlds × 3 seeds × 2 strata; 90 rows
- Aligned: Mirror NRMSE 0.00041 at 5,925 B vs untied 34,409 B and packed rank-2 10,209 B
- Random rank-2: Mirror NRMSE 0.238; private rank-2 restores exact outputs at 10,209 B
- Native PSP: NRMSE about 2.62 after charging only deterministic context seed
- Verification: pending final commit
