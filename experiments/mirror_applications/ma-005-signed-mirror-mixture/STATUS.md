# MA-005 status

- Status: PROMISING (aligned synthetic storage/quality; substantial eager CPU runtime regression)
- Branch: `research/ma-005-signed-mirror-mixture-20261007`
- Base commit: `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`
- Development complete: yes; selected LR 0.01
- Fresh/audit opened: yes; worlds 50001–50003
- Results committed: yes (`de68ec4c440e54fc4870734e9bbd1fdc99515628`)
- Verification committed: yes (`de68ec4c440e54fc4870734e9bbd1fdc99515628`)
- Registry row updated: yes (tracker commit pending)

## H / T / D / C / U

- **H:** four signed expert views of one shared matrix reproduce an aligned teacher with fewer actual bytes; arbitrary independent expert functions need private parameters.
- **T:** 16D-to-12D synthetic signed mixture, five methods, aligned and independent modes, 1,200 updates, LR 0.01, fresh worlds 50001–50003.
- **D:** PROMISING for aligned quality/storage (3/3; -41.1% bytes), but CPU runtime regressed sharply and independent expert behavior was not recovered.
- **C:** teacher was generated from the exact Givens family; the result may be an inductive-bias match.
- **U:** nonlinear/language tasks, broader capacity near convergence, learned routing, GPU/optimized kernels remain untested.

## Evidence

**Fact:** 30 fresh rows replayed, payload bytes exact; maximum MSE delta 3.3e-10; tests 4 passed.
**Interpretation:** structured functional diversity can share the matrix, arbitrary independent diversity cannot.
**Hypothesis:** optimized kernels could recover runtime without losing the storage benefit; untested.
