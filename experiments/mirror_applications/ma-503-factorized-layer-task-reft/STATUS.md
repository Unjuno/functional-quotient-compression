# MA-503 status

- Status: PROMISING (aligned synthetic orbit only; no general ReFT result)
- Branch: `research/ma-503-factorized-layer-task-reft-20261009`
- Base commit: `49e86eb0`
- Protocol frozen before fresh: yes (`8701d3b2`)
- Development complete: yes; 400 common optimizer steps selected
- Fresh/audit opened: yes; A1 timing-only rerun across 3 worlds × 3 seeds × 2 residual regimes
- Results committed: yes (`96b6cad5`)
- Verification committed: yes
- Registry row updated: yes after verification; artifact replay test passed (1/1)

## Decision

**H:** A layer-specific Mirror View plus task code compresses aligned interventions and generalizes to held-out layer-task pairs.

**T:** Synthetic 8-layer × 32-task rank-4 intervention bank in a shared 64D basis; 20% pair holdout; independent, direct LoReFT table, shared task tie, diagonal factor, generic full layer matrix, and Mirror Givens controls; `rho=0` and `.1`; 3 fresh worlds × 3 seeds.

**D:** PROMISING at `rho=0`: Mirror held-out NRMSE mean 1.6e-6 (max 1.21e-5), 3,617B vs direct coefficient table 6,949B and generic full layer matrix 4,065B. Mirror is 12.4% smaller than generic with comparable quality, but 0.063 ms CPU decode vs 0.020 ms generic. At `rho=.1`, Mirror error is .171–.201 and private pair-specific information is needed.

**C:** Teacher functions are generated from the same Givens family; generic full matrices also fit them. This does not establish a natural-task or pretrained-model gain.

**U:** Natural ReFT/LoReFT tasks, learned-basis costs, and sparse/private residual allocation are untested.

## Next action

Continue to MA-504 on its own dedicated research branch.

## Blockers

None.

## Decisions / rulings

- A1 changed only timing measurement to include View/code generation. The initial A0 metrics/timing rows are preserved; A1 is authoritative for runtime.
