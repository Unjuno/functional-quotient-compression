# MA-403 status

- Status: **FAIL** (aligned synthetic token-classification screen)
- Branch: `research/ma-403-tokenwise-mirror-modulation-20261008`
- Protocol frozen before development: yes; selected LR 0.01 on worlds 40300/40301
- Fresh worlds 40310/40311/40312 evaluated once at frozen settings
- Results: 35 rows with all methods, worlds, and development rates
- Verification: 3 tests pass; payload hashes, exact serialization, metrics replay checked
- Next candidate: MA-405 (StyleGAN2-like FFN weight modulation)

## H — hypothesis

An input/context-conditioned Givens generator produces useful token-specific views on a shared classifier, approaching an aligned teacher and improving on static context code while beating a matched FiLM generator after paying generator bytes and compute.

## T — execution

Synthetic teacher-forced token classification: length-8 streams with four contexts varying by token; input 16, hidden 32, four classes. Teacher uses one shared random ReLU base and four context-specific Givens angles. 1,200 AdamW updates, batch 128. Development worlds 40300/40301 chose LR 0.01; fresh worlds 40310/40311/40312. Controls: shared, token-input Givens generator, same-input FiLM generator, static per-context angle table, independent context classifiers.

## D — decision

**FAIL.** Fresh accuracy: shared 83.48%, Mirror generator 87.92%, FiLM generator 86.43%, static angle table 89.17%, independent bank 86.15%. The oracle teacher has 100% by construction, so Mirror remains 12.08pp below it. Mirror generator payload is 6,037B; static table is 5,209B and needs zero per-token generator MAC, while achieving higher accuracy. FiLM is 6,613B, 1.49 points below Mirror. The positive result versus the learned FiLM generator does not overcome the simpler static-code control.

## C — strongest counter-hypothesis

The context has only four discrete values, so storing one angle tuple per context is a better representation than regenerating angles from token features. Token-wise input conditioning adds compute and parameters without useful variation beyond context identity.

## U — unresolved

No natural token sequence, language model, unknown-context generalization, energy, or device latency. Independent control does not match oracle teacher and is not a capacity bound.

## Evidence separation

- **Fact:** 35 rows replay with max metric difference 5e-9; payload hashes match; round-trip logits are exact.
- **Interpretation:** dynamic generation beats the specific FiLM hypernetwork but loses to a static code table for a four-context task.
- **Hypothesis:** token-content-dependent tasks with many or continuous contexts may need dynamic views; test separately with a frozen natural or continuous-context task.
