# MA-809 status

- Status: FAIL for the frozen quality/storage frontier and Mirror-specific attribution.
- Branch: `research/ma-809-private-skill-residual-20261111`
- Base: `796310e0`
- Development: synthetic seed 80901; 4 skills; 3 common plus 1 rank-1 exception.
- Fresh/audit: sealed; development byte and native-control gates failed.
- Verification: focused test passed; exact serialized replay error is zero.

## H / T / D / C / U

- **H:** Shared Views plus one sparse private rank-1 residual recover an exceptional skill with ≥25% fewer complete bytes than independent policies.
- **T:** 8D synthetic one-step maps; 256 fitting and 512 heldout examples per skill; three common Givens-orbit maps, one rank-1 exception, heldout orthogonal input shift; six storage/control variants.
- **D:** **FAIL.** Shared-only nRMSE .03271 / success .812; private residual restores nRMSE 8.36e-8 / success 1.0 but costs 1,748 B versus 1,075 B independent. Native private control costs 1,750 B and matches quality/compute.
- **C:** Standard Givens composition plus native rank-1 residual explains the full behavior; residual metadata erases storage savings.
- **U:** Natural robot tasks, learned adaptation, fresh seeds and physical latency.
