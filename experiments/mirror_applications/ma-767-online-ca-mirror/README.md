# MA-767 — Online CA task adaptation in Mirror-code space

## H — hypothesis

Online updates to a compact rule-view code may adapt one shared cellular-automaton local rule to many tasks, preserve earlier tasks, and use fewer bytes than full-rule snapshots. Independent rules test the boundary.

## T — execution

A synthetic 1D binary local-rule prediction screen with 8 neighborhoods. Four development rules estimated the shared rank-2 basis. Each fresh seed (76711–76713) had 32 in-subspace and 32 independent rules, each with 256 online observations and 512 held-out observations. Methods: frozen shared rule, Mirror 2D code with per-task bank, full 8-logit online adaptation with per-task rule bank, and one overwritten full-rule state. The initial task-per-key serialization was retained; the final result uses compact contiguous code/rule banks for both adaptive methods.

## D — FAIL

After 64 online examples, easy-rule Mirror accuracy was within 5 percentage points of the full-rule bank in all three seeds, the Mirror bank used 808 B versus 2,128 B for full-rule snapshots (38.0%), and per-task code retention was exact. However, the frozen shared rule already averaged 0.7294 accuracy; Mirror averaged 0.7302 after adaptation. This negligible difference shows no useful quality gain from m on this screen.

For hard independent rules, Mirror averaged 0.7040 accuracy at 64 examples. Full-rule adaptation reached 0.7566 at 256 examples, compared with 0.7124 for Mirror. This indicates that full local-rule state can improve the hard stratum when given more observations. The one-state overwrite control ended at the same aggregate accuracy as frozen shared, so this sampled classification metric did not expose measurable forgetting.

## C — strongest counter-hypothesis

The 8-neighborhood binary accuracy metric is insensitive to modest rule-logit changes, and the easy rules were already classified adequately by the shared base. Therefore stored task codes may preserve distinct logits without producing useful behavior changes in this screen.

## U — unconfirmed

No full spatial pattern-generation, autonomous self-organization, natural task stream, or richer state alphabet was tested. The online result says nothing about broader CA dynamics. The hard-rule gain required full state and more observations; private adaptation efficiency remains unresolved.

## Fact / Interpretation / Hypothesis

- Fact: Mirror code-bank payload was 808 B versus 2,128 B for full-rule snapshots, but average accuracy barely changed from frozen shared; full rules improved hard-task accuracy with more observations.
- Interpretation: state compression alone did not yield a useful task-quality gain under this metric; easy task codes were functionally unnecessary, while hard tasks benefited from private state.
- Hypothesis: a richer rollout-level objective may reveal useful retained behaviors, or show that compact rule coordinates are redundant beyond local classification.
