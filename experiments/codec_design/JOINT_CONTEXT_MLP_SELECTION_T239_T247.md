# Joint-context MLP selection and passage-cluster inference (T239–T247)

This note records durable methodology conclusions from local experiments T239–T247. All functional results in this series are synthetic. Some runs use the actual TinyStories-28M MLP tensor dimensions and the executable semantic MLP codec, but no TinyStories checkpoint was available and no REAL_MODEL quality claim is made.

## 1. MLP-local ranking is not a safe whole-codec ranking

T239 evaluated semantic K128/K256/K512/K1024 MLP candidates under synthetic upstream-hidden and downstream-projection perturbations that were common to every MLP candidate.

Across eight actual-shape structured worlds and three perturbation levels (24 joint-context rows):

- local-development winner matched the joint-audit oracle in 21/24 = 87.5%;
- mean relative joint-audit regret from retaining the local MLP winner was 0.3527%;
- maximum regret was 5.3247%.

The failure rows demonstrate that a candidate selected with the rest of the model uncompressed can change rank after other codec errors are present. Therefore final MLP K selection should be performed with the concrete decoded state of the other major codec components active when feasible.

T247 further held the MLP candidates fixed while changing only the synthetic other-codec perturbation realization. In low-concentration boundary worlds the local K128 winner matched the joint oracle only 58.3% and 66.7% of 12 perturbation realizations; maximum relative regret reached 20.36% in one world. High-concentration K1024 worlds were much more stable.

## 2. Support pricing should also be checked in joint context

T240 re-priced first-order private-neuron support with the synthetic upstream/downstream perturbations already active.

Mean isolated-versus-joint support overlap was about 86% for K256/K512/K1024. Joint-aware support reduced joint-audit KL in:

- K256: 4/4 tested worlds, mean delta -0.01628;
- K512: 4/4, mean delta -0.01700;
- K1024: 3/4, mean delta -0.00864.

This does not make joint-aware pricing an oracle for K selection; one tested high-concentration world changed the development winner from K1024 to K512 while audit still preferred K1024. Support pricing and K selection remain separate stages.

## 3. No single K is a certified sentinel

T241 audited candidate-pruning ideas.

On 19 prior actual-shape synthetic worlds, using K256 or K512 as a sentinel and stopping at K128 when that sentinel lost caused a false stop in a world where K1024 still beat K128.

On the 24 T239 joint-context rows:

- omitting K256 missed the audit oracle in 3/24 rows;
- omitting K1024 missed the oracle in 12/24 rows and produced up to 13.38% relative regret.

A reduced candidate set may be useful after real development evidence exists, but **K128/K256/K512/K1024 should not be hard-pruned a priori from a single hierarchy candidate**.

## 4. Development effort should be uncertainty-adaptive

T242 varied synthetic development sample counts per layer. Across six actual-shape worlds, audit-oracle match improved from 83.6% at 8 samples/layer to 94.5% at 128; only 256 reached 100% in all six worlds. Easy worlds stabilized much earlier, while an ambiguous low-concentration structured world remained unstable at 128.

T243 showed that a raw best-versus-second relative margin is not a reliable early-stop statistic: small samples can produce spuriously large margins and nontrivial audit regret.

T245 replaced the margin with paired per-sample KL confidence bounds, using the same permutation bank across policies and Bonferroni correction over six looks and three competitors. In three boundary worlds x 96 repetitions, both tested family-alpha policies selected the audit oracle in every repetition. Clear worlds stopped early; the ambiguous world refused to certify and consumed the full development budget.

This is a fixed-world synthetic result, not a language-passage generalization theorem.

## 5. Passage, not token/layer count, is the inference unit

T246 stress-tested sequential confidence when multiple token/layer observations share a passage-level random effect.

Under a true zero candidate difference, treating the 64 within-passage observations as iid produced false winner-certification rates of:

- rho=0.25: 60.2%;
- rho=0.50: 67.0%;
- rho=0.75: 71.2%;
- rho=0.90: 71.5%.

Aggregating by passage kept the false-certification rate near 4% under the same family-alpha procedure.

Therefore **real FQC candidate-selection uncertainty must use independent passages (or another genuinely independent evaluation unit) as clusters**. Token count, layer count, attention head count, or intervention count must not inflate the effective sample size.

## Updated real-model selection protocol

1. Verify checkpoint provenance, architecture, and baseline forward.
2. Freeze calibration, development, and final-audit passage IDs before codec selection.
3. Build the other major codec components far enough that a concrete decoded joint context exists.
4. Price MLP private support in that joint context when computationally feasible; keep isolated pricing only as a diagnostic comparison.
5. Keep K128/K256/K512/K1024 in the development tournament unless real development evidence justifies narrowing the set.
6. Replay candidate MLP codecs with the actual decoded state of the other codec components active. Do not commit an MLP K solely from isolated-module replay when joint replay is available.
7. Treat independent passage as the uncertainty unit. Do not count tokens/layers as independent replicates.
8. If development cost must be adaptive, use a predeclared confidence procedure rather than a raw winner margin.
9. If candidates remain statistically unresolved, spend more passages or report uncertainty instead of forcing an early winner.
10. Run final audit only after the candidate and joint codec context are frozen.
11. Whole-model success still requires exact emitted bytes plus joint decoded replay; component metrics are not additive evidence.

## Evidence boundary

No real TinyStories weights were available during T239–T247. These experiments refine the selection protocol and expose interaction/pseudoreplication failure modes; they do not establish real 28M MLP quality or 64x whole-model success.