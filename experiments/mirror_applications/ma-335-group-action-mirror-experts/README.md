# MA-335 — C4 group-action Mirror expert views

Status: **FAIL for Mirror-specific advantage; group-structured compression demonstrated in a synthetic orbit**  
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE  
Dedicated branch: `research/ma-335-group-action-mirror-experts-20261008`  
Prior art: PA46, LieTransformer / group equivariance.

## H — Hypothesis

A compact C4 group address applied to one shared non-equivariant 2x2 linear expert can reproduce its related conjugate views, using less serialized state than independent experts. The four addresses may represent fewer than four distinct functions because the conjugation action has a stabilizer. A task outside that orbit should require private state. The group-action code must also be compared with an ordinary direct coefficient representation.

## T — Test

For each world, an analytic 2x2 teacher matrix generated four C4 conjugates and one unrelated fifth matrix. We evaluated 4,096 held-out Gaussian inputs. Methods were independent experts, hard tying, exact equivariant projection, group-action addresses with and without a private fifth expert, explicit direct matrices, and a direct C4 irreducible-coefficient control. Payloads are deterministic ZIP/NPY archives reloaded before evaluation; all arrays, compact addresses and metadata are included in actual bytes. Dev seeds were 33501/33502. Confirmatory fresh seeds were locked to 33520/33521/33522 after two documented protocol amendments; earlier seeds 33511–33519 are archived as exploratory and excluded from the verdict.

The scenario is analytic: no optimizer updates. Each method evaluated the same 4,096 examples per seed; linear evaluation work was 163,840 MACs for five tasks. Group views add 16 small 2x2 transform MACs per related view. Serialization plus evaluation wall time is recorded in `artifacts/results.csv`; this tiny CPU microbenchmark is not a deployment throughput result.

## D — Decision

**FAIL for Mirror-specific value.** Across all three confirmatory worlds, group-action payload averaged 717.3B and direct irreducible-coefficient payload averaged 722.0B (0.65% larger), with both at zero related-task nMSE and zero unrelated-task nMSE when private state was present. The simple direct control is within the frozen 10% margin, so the byte reduction cannot be attributed to Mirror specifically. Both structured methods were about 34% smaller than five independent serialized matrices (mean 1,091.3B) in this small orbit.

With no private fifth expert, group-action payload averaged 533.3B, related-view max nMSE remained zero, and unrelated-task mean nMSE was 4.6976. This is a concrete shared/private frontier in the synthetic setup. The four C4 addresses yielded only two numerically distinct related functions, consistent with the conjugation stabilizer. Exact group covariance error was zero for the reconstructed group views. The exact equivariant projection had one function and lost quality on the non-equivariant teacher orbit.

## Fact / Interpretation / Hypothesis

**Fact:** Fresh payload, quality, and function-count measurements are in `artifacts/results.csv`; hashes/bytes/metrics replayed exactly. 4/4 tests pass.

**Interpretation:** Known group structure compresses this task orbit, but a direct non-Mirror coefficient code matches the storage and quality. Address count overstates function multiplicity, and an off-orbit task needs private state.

**Hypothesis:** Group-action codes may help when task families have known symmetry and the group action can be implemented cheaply. This experiment does not show learned specialization, trained MoE behavior, language-model quality, or a Mirror-specific gain.

## C — Strongest counter-hypothesis

The apparent compression is fully explained by an ordinary irreducible-coefficient factorization of a tiny conjugation orbit. The group-action layer adds no measurable byte advantage over that simpler control and adds transform operations.

## U — Unconfirmed

No learned router or expert, realistic dimensional scale, noisy/off-orbit distribution, training efficiency, GPU throughput, or natural-language task was tested. The exact orbit and teacher matrices are analytically planted.

## Reproduction

```bash
python experiments/mirror_applications/ma-335-group-action-mirror-experts/source/run.py
python -m pytest -q experiments/mirror_applications/ma-335-group-action-mirror-experts/tests
```

Protocol amendments and excluded exploratory rows are retained in `PROTOCOL.json` and `artifacts/`.
