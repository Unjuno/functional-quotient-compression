# GOAL — Systematic Mirror Application Validation

## Objective

Systematically test the MA registry to discover where the extra low-description Mirror/View functional parameter `m` can be inserted into existing methods to replace physical parameter duplication or add useful logical functional freedom at worthwhile marginal cost.

The central program is **broad integration and falsification of `m` across strong existing methods**. Quotient/manifold discovery is a helper for finding better parameterizations or insertion points for `m`, not a replacement research objective.

The goal is **not** to prove Mirror works everywhere. Negative results are first-class outputs.

## Authoritative inputs

Read in order:
1. `AGENTS.md`
2. `WORKER_START_HERE.md`
3. `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
4. `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`
5. `docs/phase2/LATEST_WORKER_FINDINGS.md`
6. `experiments/mirror_applications/CONTEXT_ROUTER.md`
7. `experiments/mirror_applications/STATUS_BOARD.md`
8. `experiments/mirror_applications/IDEA_REGISTRY.csv`
9. selected row's prior-art references from `docs/phase2/MIRROR_APPLICATION_PRIOR_ART.md`
10. `experiments/mirror_applications/EXPERIMENT_CONTRACT.md`
11. `experiments/mirror_applications/TEMPLATE/`

## Iteration loop

Repeat:

1. Select the next UNTESTED candidate from STATUS_BOARD / WORKER_QUEUE.
   - Interpret every candidate through the doctrine: identify the exact native method/interface receiving `m` and the marginal cost/benefit being tested.
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
- Use stable MA IDs; never recycle them. MA IDs are decimal of variable length (MA-001 through MA-1045+); do not assume a fixed three-digit regex or `MA-xxx` slice. Run the read-only `experiments/mirror_applications/check_registry_integrity.py` after updating registry/claim/status documents.
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


## Measurable completion criterion

Complete a verified, scoped disposition for every candidate registered in `experiments/mirror_applications/IDEA_REGISTRY.csv`. Valid completed dispositions are PROMISING, REPLICATED, ADOPTED, FAIL, or NOT ESTABLISHED; UNTESTED and unexplained SCREENING do not count. The live registry currently has 1,155 candidates.

Each result must identify the native physical object, the exact Mirror insertion point and paid coordinate bytes, quality, active compute, training cost, measured runtime where relevant, private residual needs, strongest native/simple controls, and the scope of any fresh replication. Logical combinations are not independent capacity. ADOPTED requires replicated Pareto improvement; it is not a quota.

At each handoff, re-fetch and reconcile live `research/ma-*` branches before selecting the next untested P0. Preserve branch ownership and negative results; do not run remote CI. MA-401 (PA63) and MA-403 (PA63/PA64) are verified development-screen FAILs; after a fresh branch fetch, MA-405 (PA65) is next. The status board and live reconciliation manifest are the operational source of truth.
