# MA-244 status

- Status: PROMISING (quality/cache frontier); strict payload-byte gate missed
- Branch: `research/ma-244-kv-role-view-20261007`
- Base commit: `5b736a0f7cfca9c3f7794005dfb70954f155ea02`
- Last verified commit: pending result commit
- Development complete: yes; selected common LR `0.003`
- Fresh/audit opened: yes; worlds 24401, 24402, 24403
- Results committed: pending
- Verification committed: pending
- Registry row updated: pending until verification is committed

## Decision

In all three deliberately role-coordinate-aligned synthetic worlds, Mirror recovered attention output at MSE `9e-12`–`2e-11`, beating hard K=V and an equal-size scalar gate and matching/beating the MHA reference at the fixed update count. It used 2,100 serialized bytes versus 2,349 for MQA and 3,885 for MHA. The preregistered 20% payload reduction versus MQA was not met (10.6% measured). Physical cache state was 128 bytes versus 256 for MQA and 1,024 for MHA.

The eager CPU implementation was slower than MQA and the scalar gate. The result is a promising quality/cache mechanism signal, not an adopted natural-language architecture.

## Next action

Commit verified files, update registry/claim ledger/status board/queue, push the MA-244 branch, then continue to MA-245.

## Blockers

None for this synthetic screen.

## Decisions / rulings

- Keep the missed serialized-payload threshold visible; do not round parameter-count savings into a bytes claim.
- Evidence applies to a teacher generated from the same Givens role/head family.
- Preserve runtime regression and fixed-update boundary.
