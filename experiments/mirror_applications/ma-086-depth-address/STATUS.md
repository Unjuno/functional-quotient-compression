# MA-086 status

- Status: FAIL at development gate
- Branch: `research/ma-086-depth-address-20261007`
- Base commit: `e0cf13997e0477e9022e4c1fcf07bd99f5075ccb`
- Last verified commit: `155231548d0c202ec1d23b708206452434a7c2b7`
- Development complete: yes; seeds 86001–86002
- Fresh/audit opened: no; byte gate missed
- Results committed: yes (`155231548d0c202ec1d23b708206452434a7c2b7`)
- Verification committed: yes (`155231548d0c202ec1d23b708206452434a7c2b7`)
- Registry row updated: yes

## H / T / D / C / U

- **H:** Group-size-2 Mirror views can recover pair-aligned depth functions with at least 35% less payload than untied layers; group-size 4 may trade quality for more compression.
- **T:** Eight 8×8 linear maps, groups of 2 and 4; hard tied, rank-1 residual, Mirror-view, and untied controls; two aligned and independent development worlds; 600 updates at LR 0.01; actual serialized payload and MAC/wall metrics.
- **D:** FAIL under the preregistered gate. Mirror2 passed quality but used 2,917B vs 3,753B untied (0.777x), missing the required ≤0.65 ratio in both worlds. Fresh stayed sealed.
- **C:** The teacher was deliberately generated from pairwise Givens views, yet the byte gate still failed. Reducing physical group count to two gave a smaller 2,405B payload but quality degraded sharply.
- **U:** Fresh generalization, private residual tradeoff, nonlinear Transformer quality, and whether a different coordinate can lower bytes enough.

**FACT:** 24/24 development rows replayed with exact serialized bytes; max MSE replay delta 4.40e-11; tests 2/2 passed. Mirror2 aligned MSE median 8.35e-13, tied2 0.0584, LoRA2 0.0117, untied 1.28e-10. Fresh worlds were not opened.

**INTERPRETATION:** Grouped Mirror views preserve quality for the pair-aligned teacher but miss the predeclared storage frontier. Group size 4 reduces bytes and loses quality.

**HYPOTHESIS:** Layer groups are useful only when each saved base amortizes the cost of per-layer coordinates; more compressed coordinates or larger repeated groups need separate tests.

## Next action

Update trackers, commit/push the negative result, then continue to MA-116.

## Blockers

None.

## Decisions / rulings

Fresh worlds remained sealed after the fixed development byte-gate failure.
