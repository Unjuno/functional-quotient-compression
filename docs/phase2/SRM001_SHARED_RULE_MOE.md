# SRM001 — Shared-rule / Sparse Mirror-MoE

Date: 2026-10-07 JST

Evidence boundary: synthetic causal Transformer mechanism test only. No natural-language or LLM claim.

## Question

Can a Transformer keep a canonical shared backbone and represent specialist FFN behavior as a sparse composition of multiple shared/Mirror rules, instead of storing many independent full FFN experts?

Primary task:
- 4 domains
- 8 common rules
- private rules/domain = 16, 32, 64
- each rule is an independent random binary truth table over 16 symbols
- each example activates 2 common + 2 private rules
- primary next-token target is the count of active rules that fire (0..4)
- 1/5 of common/private rule combinations are withheld from training for recombination audit

## Main results

### Multiple selection is essential

private=32, same world, shared low-rank rule bank, 500 updates:

| active rules k | held-out acc | retained95 / 128 |
|---:|---:|---:|
| 1 | 65.59% | 0 |
| 2 | 72.75% | 2 |
| 4 | 99.84% | 128 |

This is not a top-1-expert effect. The gain requires composing the multiple rules that are active in the example.

### Shared low-rank rule atoms vs byte-matched standard MoE

private=32, 4 worlds, 500 updates:

- standard full-FFN MoE E=9: median 215,343 B; held-out 66.93%; retained95 median 0/128
- shared low-rank top-4: median 208,287 B; held-out 99.87%; retained95 median 128/128

Direction matched in 4/4 worlds.

### Mirror-specific residual

Mirror rules do not store independent adapter matrices. They share W1/W2 and store only structured stretch+shear coordinates around the shared GELU.

private=32 rank sweep, same world, 1000 updates:

| rank | bytes | held-out | retained95 /128 |
|---:|---:|---:|---:|
| 4 | 73,110 | 91.63% | 51 |
| 8 | 77,462 | 95.70% | 94 |
| 16 | 86,166 | 99.15% | 126 |

With rank16 fixed over 4 worlds:
- standard MoE E=2: median 88,215 B; held-out 82.89%; retained95 median 8.5/128
- Mirror: median 86,166 B; held-out 99.20%; retained95 median 126/128

Mirror won 4/4 worlds.

### Stronger learned standard-MoE routing control

To rule out fixed hash routing as the explanation, a learned soft router was allowed to mix all full experts.

- private=32, E=2: 87,811 B; held-out 84.93%; retained95 23/128
- Mirror rank16: 86,166 B; held-out 99.15%; retained95 126/128

- private=64, E=3: 127,527 B; held-out 67.22%; retained95 0/256
- Mirror rank16: 123,926 B; held-out 99.15%; retained95 251/256

The positive signal is therefore not explained only by the deterministic rule-to-expert assignment.

### Full independent expert upper control

If one independent full FFN is stored per rule, standard MoE reaches approximately 99.97% held-out at 500 updates, but storage rises to:
- private=32 / 136 rules: 2,332,093 B
- private=64 / 264 rules: 4,499,773 B

This does not prove a general 27x–36x compression factor. The synthetic task is explicitly factorized into reusable rules. It does show that the same family can trade large independent expert storage for a much smaller shared FFN + rule-coordinate representation.

### Router bridge

A supervised router learned the explicit rule-token -> Mirror-address mapping:
- routing position accuracy: 100%
- all four addresses correct: 100%
- private=32 held-out after 1000 updates: ~99.5%
- serialized bytes: 104,690 B

A same-size standard MoE E=3 was 106,367 B and held-out 84.31%.

This is only a bridge. It uses explicit rule tokens and routing supervision; it is not emergent natural-language routing.

## Important negative result: credit assignment

Changing the target from count to parity of the four rule firings caused both standard MoE and shared/Mirror approaches to stay near chance (~50%). Adding a shared nonlinear MLP after the specialized FFN did not fix it. Pretraining the rule bank on count and then finetuning parity also failed.

Interpretation: a compositional parameterization is not enough. The training loss must provide usable credit to the underlying rule factors. This is the primary unresolved risk for language-model application.

## Decision

- Sparse multi-rule composition: PASS on the decomposable synthetic task.
- Fixed-byte shared-rule capacity signal: PASS.
- Mirror-specific low-description residual signal: PASS.
- Standard learned-router adversary: Mirror signal survives.
- Full independent experts: recover quality with far larger storage.
- Emergent routing from LM loss: NOT TESTED.
- Non-decomposable credit assignment: FAIL in the parity adversary.
- Natural-language LLM claim: NOT ESTABLISHED.

Next gate: construct intermediate compositional targets between count and parity, then move supervised routing toward weakly supervised and LM-loss-only routing before any natural-language claim.
