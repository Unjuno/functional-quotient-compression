# MA-567 status

- Status: **FAIL for incremental Mirror-specific benefit; actual MoD application remains NOT ESTABLISHED**
- Branch: `research/ma-567-mod-mirror-routing-20261102`
- Protocol freeze: `c8d240bc`; A1–A6 amendments individually preserved; A6 is the audited replay.
- A6 reuses the preregistered seed IDs after measurement corrections, so it is an audited replay, not an independent fresh replication.

## H / T / D / C / U

**H:** At identical MoD route allocation, a role-conditioned view recovers functions missed by routing alone; test whether Mirror improves quality/bytes/compute over direct-angle and rank-2 controls.

**T:** Shared two-block, 8-channel synthetic model; 512 tokens per seed, balanced roles, top-50% difficulty routing; MoD only, Mirror Givens, direct role-angle table, generic rank-2, dense upper. A6 uses the original float64 inputs/shared weights and saved/loaded float32 view tables; teacher phase is independently computed in high precision. Shared/router/view/basis metadata are serialized and decoded. End-to-end CPU timing executes the network and routing; A0–A5 are retained as superseded audits.

**D — FAIL for incremental Mirror benefit.** A6 fresh-seed replay mean NRMSE: MoD `0.1346`, Mirror/direct-angle/rank-2 `3.52e-09`. The direct scalar-angle table matches Mirror exactly in outputs, bytes (`3204 B`) and active MACs (`83968`) in all worlds. This is an exact generic reparameterization, so the Mirror-specific claim fails. The application to a trained MoD Transformer is NOT ESTABLISHED.

**C:** The synthetic teacher explicitly uses two role-conditioned Givens angles; Mirror and direct-angle controls encode the same state and execute the same operation. The task has no learned router, language model or learned role code.

**U:** Trained MoD/Transformer and language NLL; non-Givens role functions; GPU/energy; deployment kernels; independently fresh replication after audit corrections.

## Facts

- Selected deep fraction is 0.5 for all methods; dense upper actually executes block 2 on 100% of tokens.
- Mean actual payload: MoD `2797 B`; Mirror/direct angle `3204 B`; rank-2 `3417 B`; dense `3403 B`.
- Active MAC proxy counts matrix and view multiply-accumulates, including both projections and executed block 2: MoD 81,920; Mirror/direct/rank-2 83,968; dense 100,352. It excludes tanh/nonlinear arithmetic; router sorting is separately logged.
- End-to-end Python CPU batch means (ms): MoD `0.0659`; Mirror `0.0824`; direct `0.0971`; rank-2 `0.1159`; dense `0.0780`. These unfused NumPy timings are implementation-sensitive.
- Astra independently regenerated the A0 teacher, replayed all 15 A6 fresh rows from saved NPZs with zero metric difference, confirmed every payload hash/size and route, and verified the exact Mirror/direct alias. A6 verifier passes. A0–A5 quality/runtime comparisons are superseded and withdrawn.

## Interpretation

A role code can restore functions MoD-only routing misses while preserving the same 50% selected-depth budget, in this constructed example. The exact direct-angle control demonstrates that this function gain has no incremental Mirror-specific mechanism. No capacity claim is supported.
