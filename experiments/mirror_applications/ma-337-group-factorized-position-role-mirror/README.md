# MA-337 — Factorized position × role group views

Status: **FAIL for Mirror-specific value; synthetic factorized group generalization demonstrated**
Dedicated branch: `research/ma-337-group-factorized-position-role-20261008`
Prior art: PA46 group equivariance; PA07 projection sharing in attention.

## H — Hypothesis

Two independently meaningful coordinates, position C4 and semantic-role C2, can compose through one shared operator to recover held-out group compositions more compactly than independent operators. A flat table should not recover unseen compositions. A native direct-coordinate control tests whether the result is Mirror-specific. A task outside the group orbit may need private state.

## T — Test

The synthetic operator is 8×8 over four positions and two roles. Each target is the conjugate of a seeded dense base operator by `kron(P^p, S^r)`, where `P` is a cyclic position shift and `S` swaps the two roles. Six of eight composition IDs are marked seen; `(2,1)` and `(3,1)` are held out. One independently sampled ninth operator is off-orbit. Each seed evaluates 2,048 held-out Gaussian vectors.

Controls: eight independent operators, nine independent operators including the off-orbit task, a flat table containing only six seen compositions, hard tying, the group-averaged exactly equivariant operator, factorized group-action coordinates, the same native direct position/role coordinates, and factorized state with/without private off-orbit state. Arrays and compact coordinates are serialized in deterministic ZIP/NPY payloads and reloaded before evaluation. Dev seeds 33701/33702; fresh seeds 33711/33712/33713. No optimizer updates; this is an analytic orbit screen.

## D — Decision

**FAIL for Mirror-specific advantage.** On all three fresh seeds, the factorized action recovered both held-out compositions at nMSE 0. Its mean payload was 974.3B versus 3,598.3B for eight independent operators, a 72.9% reduction. It produced eight distinct functions from the eight position-role pairs. The flat seen-only table could not represent the two held-out IDs.

The native direct-coordinate control also had zero held-out error and used 985.3B on average, only 1.1% more than the Mirror encoding. The storage/generalization effect is therefore explained by ordinary factorized group coordinates; it is not a Mirror-specific gain. Without private state, the off-orbit ninth task had mean nMSE 1.0958. Adding a private operator restored zero error at 1,411.3B versus 4,036.3B for nine independent operators. Hard tying and the exactly equivariant projection each retained only one function and had high error on held-out task operators.

## Fact / Interpretation / Hypothesis

**Fact:** Fresh results, serialized byte counts, hashes and metrics are in `artifacts/results.csv`; replay matched exactly. Three tests pass.

**Interpretation:** Factored coordinates generalized to held-out compositions because the group generators were known. Mirror syntax added no useful advantage over ordinary direct position/role codes. Off-orbit functions require private state.

**Hypothesis:** Larger attention systems may benefit from explicit factorized group conditioning when composition structure is known, but any advantage should first be credited to group factorization and benchmarked against ordinary coordinates.

## C — Strongest counter-hypothesis

The result is a deterministic consequence of a planted exact group orbit. The ordinary direct-coordinate code recovers it equally well, and no learned attention, router, or natural position distribution was involved.

## U — Unconfirmed

No trained Transformer, attention masking, unseen group generator, learning efficiency, GPU inference, or language-model quality was tested. The held-out task IDs are unseen compositions, but their group generators are known by design.

## Reproduction

```bash
python experiments/mirror_applications/ma-337-group-factorized-position-role-mirror/source/run.py
python -m pytest -q experiments/mirror_applications/ma-337-group-factorized-position-role-mirror/tests
python experiments/mirror_applications/ma-337-group-factorized-position-role-mirror/source/verify.py
```
