# MA-818 status

- Status: FAIL (Mirror-specific); narrow mechanism PASS
- Branch: `research/ma-818-mirror-new-operation-codes-20261008`
- Base commit: `28194ea` (`research/mirror-application-worker-ready-20261007`)
- Last verified commit: `c448539`
- Development complete: yes
- Fresh/audit opened: yes, after source/protocol hashes were frozen
- Results committed: yes
- Verification committed: yes
- Registry row updated: yes

## Next action

Push the checked MA-818 branch, then refresh live status and draw a new eligible P0 candidate uniformly at random.

## Blockers

None.

## Decisions / rulings

Record deviations from the original protocol here.

- In response to development results, explicitly included the unrelated private-operator fit threshold in the success gate before freezing the fresh run; no method, seed, support size, or hyperparameter changed.
- Protocol JSON syntax was corrected before fresh access; fresh data was regenerated from the frozen source/protocol run.
- Mirror-specific gate failed: ordinary structured operation embedding matched actual bytes, payload SHA-256, and quality exactly.
