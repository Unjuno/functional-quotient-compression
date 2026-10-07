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

An adversarial 16-class bitmask task outputs the four individual firing bits, so simple count/addition is insufficient.

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

A longer standard-MoE run on private=32 reached:
- 500 updates: ~66.9%
- 1000 updates: ~84.0%
- 1500 updates: 92.81%

Therefore the strongest current claim is a large learning-efficiency gap at fixed updates/tokens. Final near-convergence capacity superiority is not yet established.

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

private=64 dev:
- standard MoE E=3: 128,767 B; held-out 67.11%; retained95 0/256
- Mirror rank16: 123,926 B; held-out 99.15%; retained95 251/256

### Full independent expert upper control

If one independent full FFN is stored per rule, standard MoE reaches approximately 99.97% held-out at 500 updates, but storage rises to:
- private=32 / 136 rules: 2,332,093 B
- private=64 / 264 rules: 4,499,773 B

This does not prove a general compression factor. The synthetic task is explicitly factorized into reusable rules. It does show that the same family can trade large independent expert storage for a much smaller shared FFN + rule-coordinate representation.

### Non-additive bitmask control

The target is the four individual rule truth values encoded as one of 16 classes, not their count.

private=16, 4 worlds, 500 updates:
- standard MoE: median 132,802 B; held-out 48.88%
- shared low-rank rules: median 133,469 B; held-out 73.29%
- shared-rule direction won 4/4 worlds

private=32 dev:
- standard MoE: 33.64%
- shared low-rank rules: 74.41%

Both methods remain far from full rule retention, so this is a learning-efficiency signal rather than a capacity result.

A separate parity target remained near chance for both standard and shared/Mirror variants, even with a post-FFN nonlinear MLP or count->parity curriculum. The current factorization is therefore not universal across arbitrary compositions.

## Routing experiments

### Supervised learned router

A learned router maps the four explicit rule-token positions to four Mirror addresses.

private=32, 4 worlds, ~same bytes, 1000 updates:
- standard MoE E=3: median 106,367 B; held-out 82.37%; retained95 median 13.5/128
- Mirror + learned 4-address router: median 104,582 B; held-out 99.56%; retained95 median 127/128

This confirms that a learned multi-address router can preserve most oracle-Mirror performance, but it uses explicit routing supervision.

### End-to-end routing from LM loss only

A deterministic straight-through router receives rule-token representations and is trained only through LM loss. Small asymmetric initialization of Mirror codes breaks the all-zero routing symmetry.

private=16, 4 worlds, 1000 updates:
- standard MoE E=2: 77,015 B; median held-out 94.86%; retained95 median 42.5/64
- end-to-end Mirror router: 75,758 B; median held-out 97.01%; retained95 median 52.5/64
- Mirror accuracy wins 3/4 worlds

private=32, 4 worlds:
- standard MoE E=3: 106,367 B; median held-out 83.33%; retained95 median 11/128
- end-to-end Mirror router: 103,086 B; median held-out 91.19%; retained95 median 50/128
- Mirror wins 4/4 worlds

private=64 dev:
- standard MoE E=5: 165,135 B; held-out 64.94%
- end-to-end Mirror router: 157,742 B; held-out 76.30%

At private=64 both models remain undertrained; only 3/256 private rules reached 95% retention for the Mirror model.

The end-to-end router becomes highly stable after training: same-rule address consistency is about 0.98-0.996 in the tested private=16/32 runs. It still uses only a subset of available addresses. Simple load-balancing regularization increased address usage but did not improve quality. The current bottleneck is therefore good address discovery, not mere load balance.

## Decision

- Sparse multi-rule composition: PASS on the decomposable synthetic task.
- Multi-selection is materially better than top-1/top-2: PASS.
- Fixed-byte shared-rule learning-efficiency signal: PASS.
- Mirror-specific low-description residual signal on count task: PASS.
- Full independent experts: recover quality with far larger storage.
- Supervised learned router: PASS.
- LM-loss-only end-to-end sparse routing: PROMISING; private=32 wins 4/4, private=16 wins 3/4, not yet a general result.
- Non-additive bitmask transfer of the advantage: PROMISING but weaker.
- Parity / hard credit assignment: FAIL.
- Near-convergence capacity superiority: NOT ESTABLISHED.
- Natural-language LLM claim: NOT ESTABLISHED.

## Next gate

Construct ordered non-commutative rule transformations, where A then B differs from B then A. Compare sequential independent full experts, sequential low-rank shared atoms, and sequential Mirror atoms under actual serialized-byte and compute frontiers.

This is the decisive test for genuine functional composition rather than additive evidence accumulation.
