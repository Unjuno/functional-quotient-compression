# MA-795 status

- Status: FAIL (scoped synthetic screen; Mirror attribution)
- Branch: `research/ma-795-merge-prune-mirror-20261111`
- Base commit: `c4990b6d`
- Development: seed 79501; one synthetic world.
- Fresh/audit: sealed and unopened after exact native low-rank alias and failed development quality gate.
- Verification: focused test passes; all serialized inference payloads roundtrip exactly.

## H / T / D / C / U

- **H:** Rank-1 Views after merge/prune recover removed expert roles at ≤.02 output nRMSE with ≥25% fewer bytes than independent experts, beating merge/prune controls without greater active compute.
- **T:** Eight 12×12 synthetic experts in two clusters; 128 train/test examples per role; independent, prune-only, merge-only, small-from-start, View and native low-rank controls; complete NPZ bytes, test nRMSE, MAC proxies, example counts, wall time, zero updates.
- **D:** **FAIL.** View reaches .02472 weighted test nRMSE and 41.5% byte savings vs independent, but fails quality and exactly aliases native low-rank. Simpler two-expert controls use fewer bytes and MACs.
- **C:** Rank-1 shared residual structure was built into the synthetic teacher, so standard low-rank factors explain the View's recovery; small private perturbations explain residual error.
- **U:** Fresh worlds, real merge/pruning, learned routing, training trajectories, neural/LM quality, GPU runtime, and private-residual frontier.

## Next action

Record the scoped FAIL in the research registry and move to the next eligible P0.
