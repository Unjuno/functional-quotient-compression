# MA-801 status

- Status: FAIL for Mirror-specific attribution and complete-byte savings.
- Branch: `research/ma-801-vector-network-task-state-20261111`
- Base commit: `56e37142`
- Development: seed 80101; checkerboard half of task-state pairs held out.
- Fresh/audit: sealed after exact native CP alias and storage gate miss.
- Verification: focused test passes; payload replay is exact.

## H / T / D / C / U

- **H:** Factorized task×state coordinates compress coefficients over rank-1 atoms and generalize heldout task-state compositions under input-local top-2 selection.
- **T:** Four tasks × four states × eight atoms; 256 examples/pair; independent, task-only, state-only, Mirror CP and native CP controls.
- **D:** **FAIL.** Near-zero heldout error and 28 active-atom subsets, but 2,269 B vs 2,275 B independent is only 0.26% savings; native CP exactly aliases.
- **C:** Ordinary CP factorization and actual entropy-coded storage explain the result.
- **U:** Trained Vector Networks, learned fast-weight state, larger dimensions, quantization, natural OOD tasks and GPU kernels.

## Next action

Verify the payloads, record the second same-cause family failure, pause MA-801 and proceed to MA-802.
