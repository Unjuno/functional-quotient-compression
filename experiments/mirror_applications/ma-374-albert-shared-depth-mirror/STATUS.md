# MA-374 status

- Status: PROMISING only for a development final-depth signal; preregistered multi-depth gate missed. Original payload replay is not independently reproducible from this branch; see the A1 runtime replay audit.
- Branch: `research/ma-374-albert-shared-depth-mirror-20261008`
- Base commit: `de5504c`
- Last verified commit: `80725b5`
- Development complete: yes (seeds 37401, 37402)
- Fresh/audit opened: no
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes (PROMISING)

## Next action

Preserve the scoped development signal and replay limitation; continue with the current queue. Fresh remains sealed.

## Blockers

The preregistered multi-depth gate failed in development. The original JSON records did not have their referenced inference payloads archived. Current-runtime same-seed regeneration yielded 0/12 matching original payload hashes and material metric drift for some controls; the regenerated payloads and exact variant replay are archived under `protocol_variants/runtime_replay_20261008/`.
