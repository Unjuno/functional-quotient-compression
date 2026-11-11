# MA-872 results: discrete hypotheses for module reuse

## H — Hypothesis

When each new task selects one of eight learned module recombination rules, a discrete address should preserve function quality with fewer support examples than continuous coordinate fitting and reduce payload relative to independent task coefficients. Mirror-specific attribution also requires beating native nearest-codebook selection at comparable cost.

## T — Trial

Each synthetic world contained eight shared nonlinear modules and eight equally spaced rule prototypes in a rank-2 coefficient plane. Twenty-four training tasks (three per prototype) estimated the prototype codebook and continuous PCA subspace. Sixteen heldout task instances selected known prototypes with fresh support and test observations. Fresh seeds were 941, 953, and 971; support counts were 1, 2, 4, 8, 16, and 32; each task had 2,048 test inputs. Controls were native nearest codebook, continuous rank-2 PCA, random prototypes, independent coefficient fit, pooled coefficients, and teacher rule.

## D — Decision

**FAIL for Mirror-specific attribution; discrete native codebook utility passes this synthetic screen.** Discrete selection first met the quality gate at 2 examples in all fresh worlds, versus 4 for continuous PCA. Its payload is 569 B versus 809 B independent (29.7% lower), but continuous PCA is smaller at 489 B. Native nearest-codebook selection exactly matches Mirror choices, outputs, and payload length.

## C — Strongest counter-hypothesis

The gains come from ordinary finite hypothesis testing over a learned coefficient codebook. The Mirror address is just a native codebook index; it has no distinct function or storage advantage. The task generator also makes the prototype set known and finite.

## U — Unconfirmed

Unseen recombination rules, online rule discovery, changing modules, larger or hierarchical codebooks, non-synthetic tasks, neural/RL adaptation, and generalization beyond the eight known prototypes remain untested. No independent-capacity claim is made from the eight addresses.

## Facts

- At 2 shots, the maximum fresh-world mean function MSE for discrete selection was 0.0000211; the worst individual heldout task across all 48 task-world instances was 0.0000325. At 1 shot, its maximum world mean was 0.1298.
- Continuous PCA first met the per-world mean MSE threshold at 4 shots (maximum world mean 0.001706); at 2 shots it was 0.2238. Independent fitting first met the per-world mean gate at 16 shots (maximum world mean 0.001130); its 8-shot fit remained unstable in this correlated module bank.
- Sixteen-task actual payloads, including module bank and metadata, were 569 B for discrete Mirror/codebook, 489 B continuous PCA, 809 B independent, 569 B random prototypes, 329 B pooled, and 809 B teacher coefficients. Discrete saves 29.7% versus independent but costs 16.4% more bytes than continuous PCA.
- Mirror and native nearest-codebook selected the same prototype for all 16 heldout tasks across all fresh seeds and support counts; per-task errors match exactly. Their payloads are both 569 B.
- At four support examples, codebook search costs 512 multiply-add proxy operations per task (8 candidates × 4 examples × 8 module outputs × multiply/add); continuous support adaptation uses a two-coordinate least-squares solve. Shared module evaluation is 72 MACs/query; direct coefficient decode is 16 MACs, rank-2 decode is 48 MACs.
- Mean measured discrete adaptation was 0.688 ms per 16-task batch at four shots, including codebook comparisons; timings are host-local.

## Interpretation

A finite rule codebook can reduce the number of examples needed to identify a known recombination rule. The codebook pays more bytes than continuous PCA but reaches the quality gate with half the support examples in this screen. This is a storage/sample-efficiency tradeoff from ordinary discrete hypothesis selection, and native codebook search duplicates it exactly.

## Hypothesis for follow-up

A future experiment could withhold combinations of previously learned “what” and “how” factors, then test whether a compact code composes to unseen rules without storing those composite prototypes. That is distinct from this known-codebook test and should compare against a native factorized decoder.
