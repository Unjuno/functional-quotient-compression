# MA-866 status

- Status: FAIL for Mirror-specific attribution; communication-quality screen passes its fixed gates
- Branch: research/ma-866-mesh-admission-policy-code-20261111
- Base commit: 3b08b52c
- Development complete: fixed deterministic schema/code; no tuning
- Fresh/audit opened: yes; seeds 503, 521, 547
- Results committed: pending
- Verification committed: pending

## Decision

One-agent admission sends 1/32 the messages and bytes with mean nMSE .058–.060 versus centralized .042, within the .35 quality gate. The 38 B Mirror program selects exactly the same agent as the 24 B native type/reliability rule, so there is no Mirror-specific gain.

## Next action

Continue with MA-867, factorized agent-role × query codes; test unseen role-query combinations.
