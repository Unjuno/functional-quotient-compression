# GOAL — Systematic Mirror Application Validation

## Objective

Systematically test the MA registry to discover where a low-description Mirror/View coordinate can replace physical parameter duplication with useful logical multiplicity.

The goal is **not** to prove Mirror works everywhere. Negative results are first-class outputs.

## Authoritative inputs

Read in order:
1. `AGENTS.md`
2. `WORKER_START_HERE.md`
3. `docs/phase2/LATEST_WORKER_FINDINGS.md`
4. `experiments/mirror_applications/CONTEXT_ROUTER.md`
5. `experiments/mirror_applications/STATUS_BOARD.md`
6. `experiments/mirror_applications/IDEA_REGISTRY.csv`
7. selected row's prior-art references from `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`
8. `experiments/mirror_applications/EXPERIMENT_CONTRACT.md`
9. `experiments/mirror_applications/TEMPLATE/`

## Iteration loop

Repeat:

1. Select the next UNTESTED candidate from STATUS_BOARD / WORKER_QUEUE.
2. Check whether an experiment directory or research branch already claims that ID.
3. Create a dedicated research branch from the latest worker-ready baseline.
4. Copy the TEMPLATE into a stable MA directory.
5. Fill README + PROTOCOL before opening fresh/audit data.
6. Implement the cheapest scientifically valid mechanism screen.
7. Run development-only selection.
8. If the mechanism clearly fails its predeclared gate:
   - record FAIL;
   - commit result + verification;
   - update registry/status board;
   - continue.
9. If development is promising:
   - lock configuration and fresh seeds/worlds;
   - run fresh replication;
   - record actual serialized bytes and compute;
   - compare the strongest simple control;
   - classify PROMISING / REPLICATED / FAIL.
10. Update registry, claim ledger, status board and current-state docs only after verification exists.
11. Continue to the next candidate.

## Candidate order

Prefer:
1. literature-derived P0 cross-over queue;
2. original P0 family queue;
3. P1 only after the relevant P0 family is understood;
4. P2 only when it answers a concrete gap found in earlier experiments.

Do not choose by novelty excitement alone.

## Family batching

Workers may share code/checkpoints/data fixtures across candidates in the same family, but every MA ID keeps:
- its own hypothesis;
- its own controls;
- its own status;
- its own result row;
- its own conclusion.

## Hard scientific rules

- Actual serialized inference bytes are authoritative for storage.
- Fixed-update wins are learning-efficiency evidence, not capacity proof.
- Combination count is not independent capacity.
- Fresh/audit data never tune hyperparameters.
- A Mirror-specific claim requires beating a simpler non-Mirror shared/low-rank/generated control.
- If an oracle router, executor, mask, target, extra forward pass, or additional compute is supplied, state it.
- Keep storage, active compute, training compute, runtime and quality as separate axes.
- Never delete or hide a negative result.

## Hard repository rules

- Never implement experiments by editing `third_party/nanoGPT/`; use it as a stable baseline.
- Never overwrite historical SRM/TM protocols.
- Never merge to main unless explicitly requested.
- Use stable MA IDs; never recycle them.
- Large checkpoints may stay outside Git, but hashes/provenance must be recorded.

## Stop / escalate conditions

Stop the autonomous loop and report instead of guessing when:
- an experiment requires unavailable hardware to answer the registered hypothesis;
- a required prior-art control cannot be implemented or reproduced;
- two consecutive candidates in one family fail for the same demonstrated structural reason;
- repository evidence conflicts materially with the registry;
- a destructive/irreversible repository action would be required.

## Success condition for the program

The program is successful when it produces a map, not only wins:

- which physical multiplicities can be replaced;
- which require private residuals;
- which Mirror coordinate families work;
- where simple tying/LoRA/hypernetworks are already sufficient;
- how storage, compute, learning speed and quality trade off.

A candidate reaches ADOPTED only after a useful Pareto improvement is replicated and its nearest simple control is beaten.
