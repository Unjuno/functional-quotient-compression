# MA-250 status

- Status: **PROMISING (aligned linear expert basis)**
- Branch: `research/ma-250-vsa-expert-address-20261007`
- Base commit: `d657dbe8539e5af756cb89533fd9651e968ef4e3`
- Development: complete; LR 0.01 selected on world 25000
- Fresh/audit: complete; worlds 25001–25003
- Results committed: yes (`05ad4f7efb92a3b7bbe8f4f0674377223d6bc768`)
- Verification committed: yes (`05ad4f7efb92a3b7bbe8f4f0674377223d6bc768`)
- Registry row updated: see status board and registry

## H / T / D / C / U

- **H:** A shared expert matrix with per-role Givens coordinates can recover four role-specific functions from one physical expert when roles share that transformed basis; compare MAP, Hadamard and HRR binding.
- **T:** 16D-to-12D linear roles; aligned and independent teacher modes; eight controls; 1,200 AdamW updates, matched minibatches; dev world 25000; fresh worlds 25001–25003; common LR 0.01.
- **D:** PROMISING on aligned mode: Mirror matched untied quality in 3/3 worlds with 3,406B vs 5,522B (-38.3%); fixed MAP/Hadamard/HRR controls had much higher MSE. Independent roles defeated all shared transforms; untied fit exactly.
- **C:** Teacher is intentionally aligned to Givens; VSA codes were fixed random codes, not optimized.
- **U:** Learned VSA codes, nonlinear experts, MoE routing, language, near-convergence frontier, and accelerator kernels.

## Verification

Four tests passed. All 48 fresh metrics replayed; maximum MSE difference 4.77e-10, R² difference 4.90e-9, exact payload-byte match 48/48.
