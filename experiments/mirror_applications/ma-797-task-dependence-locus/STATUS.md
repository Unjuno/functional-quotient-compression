# MA-797 status

- Status: FAIL for Mirror-specific attribution; synthetic locus mechanism passes within scope.
- Branch: `research/ma-797-task-dependence-locus-20261111`
- Base commit: `9d54e528`
- Development: seed 79701; three heldout +1 task/locus combinations.
- Fresh/audit: sealed after exact native low-rank alias.
- Verification: focused test passes; compressed payload roundtrips are exact.

## H / T / D / C / U

- **H:** Factorized task and locus codes recover heldout functions, identify the generating locus, and reduce full storage.
- **T:** Synthetic 8D three-stage system; input/recurrent/readout loci, factor -1/0/+1; positive tasks held out; independent, shared-only, View and native low-rank controls.
- **D:** **FAIL** Mirror attribution. View reaches 4.5e-8 heldout nRMSE at 2,018 B (30.1% below independent); wrong-locus intervention yields .0281 error. Native low-rank matches payload/function exactly and the rank-1 path has extra MACs.
- **C:** This is ordinary conditional low-rank stage modulation with a categorical locus selector.
- **U:** Learned locus inference, neural training, natural tasks, fresh seeds and deployment kernels.

## Next action

Pause the related latent-dynamics task-code family after the second exact native alias; proceed to unrelated MA-800.
