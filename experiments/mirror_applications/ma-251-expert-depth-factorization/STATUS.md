# MA-251 status

- Status: **PROMISING (factorized linear role-depth mechanism)**
- Branch: `research/ma-251-expert-depth-factorization-20261007`
- Base commit: `809f9d50ef85d470d8bac045db291349e264dcba`
- Development: complete; LR 0.01 selected on world 25100
- Fresh/audit: complete; worlds 25101–25103
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending

## H / T / D / C / U

- **H:** Separate expert and depth Givens codes can recover a Cartesian set of role-depth functions from one matrix when transformations factor across the axes.
- **T:** 4 experts x 4 depth positions, 16D-to-12D regression; aligned-factorized and independent-pair teachers; nine controls; 1,200 updates; fresh worlds 25101–25103.
- **D:** PROMISING: factorized Mirror matched untied quality 3/3 aligned worlds with 3,608B vs 15,317B untied and 3,803B Cartesian table. Independent pair functions defeated factorized and Cartesian Mirror; full untied remained accurate.
- **C:** Teacher was deliberately factorized and matched the Givens mechanism.
- **U:** Natural language, router behavior, nonseparable interactions, private residual rank, fixed-byte near-convergence capacity, optimized kernels.

## Verification

Four tests passed; all 54 fresh rows replayed; maximum MSE difference 4.31e-10, R² difference 4.66e-9, exact payload-byte match 54/54.
