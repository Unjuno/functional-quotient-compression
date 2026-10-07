# TM001 — Parallel Period Token Mixing

Date: 2026-10-07 JST

Evidence boundary: synthetic tiny causal Transformer and CPU microbenchmark only. No natural-language, GPU, equal-FLOP, or production-speed claim.

## Question

Can a period of P future tokens be emitted in one Transformer forward by representing future positions as parallel phase slots?

## Setup

32 states, 12 rules, two random state permutations per rule. A packet uses one branch for all P future states.

Three conditions:
- deterministic: branch 0 is fixed;
- revealed: branch is random but included in the context;
- hidden: branch is sampled once per packet and omitted from context.

Models share a 2-layer width-32 Transformer core with 4 heads and FFN width 64.

Comparisons:
- AR: ordinary autoregressive Transformer with KV-cached inference;
- Direct period: P learned phase slots see the past context but not each other;
- Triangular mixer: P learned phase slots are evaluated in one tensor; later slots may attend earlier latent slots, never future target-token embeddings.

Training: AdamW, 1600 updates, batch 128, FP32. Storage claims use actual deterministic serialized payload bytes.

## Positive result

When the whole future packet is determined by information available at macro-step start, one-forward P-token generation works.

P=4 fresh worlds 201/202/203:
- deterministic Direct: joint accuracy 100% in 3/3;
- deterministic Mixer: joint accuracy 100% in 3/3;
- revealed Direct: joint accuracy 100% in 3/3;
- revealed Mixer: 100%, 100%, 99.87%.

World 201 deterministic sweep:
- P=2: AR / Direct / Mixer all 100% joint;
- P=4: all 100%;
- P=8: all 100%.

At P=4 the payloads were approximately 84,772 B AR versus 85,376 B Direct and 85,375 B Mixer.

## Mixer-specific result

No stable quality advantage from triangular slot mixing was observed over Direct period slots. The positive mechanism in this fixture is parallel horizon/phase queries, not the triangular mixer itself.

## Hidden information boundary

P=4, 3 fresh worlds, median sequence NLL:
- KV-cached AR: 0.7686 nat; valid generated trajectory 100%;
- Direct period: 3.3181 nat; valid trajectory 40.9%;
- Triangular mixer: 3.3544 nat; valid trajectory 39.3%.

The two complete packet trajectories are distinct in these worlds, so the packet-level latent entropy is ln(2)=0.6931 nat. AR pays this uncertainty once and conditions later predictions on the realized previous token. Factorized period slots have no single sampled packet latent, so uncertainty is duplicated across slots.

World 201 period sweep:

| P | AR NLL | Direct NLL | Mixer NLL | AR valid | Direct valid | Mixer valid |
|---:|---:|---:|---:|---:|---:|---:|
|2|0.763|1.480|1.484|1.000|0.792|0.766|
|4|0.765|3.318|3.256|1.000|0.414|0.424|
|8|0.777|10.488|10.769|1.000|0.029|0.029|

Triangular latent attention did not solve this joint-distribution problem.

## Packet-latent diagnostic

A two-component packet latent shared across all P slots improved the hidden P=4 result in one fresh world but did not close the gap:
- factorized Mixer: 3.256 nat;
- packet-latent mixture, 2000 updates: 1.853 nat, both-mode coverage 65.4%;
- hard-EM variant, 1600 updates: 2.646 nat;
- AR: 0.765 nat.

This lane is promising but not stable enough to pass.

## CPU inference benchmark

PyTorch 2.10.0+cpu, one thread. Both methods process the same 4-token context once per benchmark invocation. AR then generates P tokens with a KV cache; period decoding emits P logits in one forward.

Throughput speedup period / cached AR:

| P | batch | Direct | Mixer |
|---:|---:|---:|---:|
|2|1|1.71x|1.65x|
|2|32|1.28x|1.19x|
|2|128|0.94x|0.91x|
|4|1|3.06x|3.13x|
|4|32|1.76x|1.66x|
|4|128|1.22x|1.19x|
|8|1|5.38x|5.10x|
|8|32|2.08x|2.06x|
|8|128|1.28x|1.29x|

The speedup is not P and shrinks with batch saturation. P=2 at batch128 is slower than cached AR.

## Decision

- one-forward P-token generation when packet information is already available: PASS;
- P=4 positive replication: PASS;
- P=8 deterministic packet: PASS on one fresh world;
- triangular mixer-specific quality advantage: FAIL / not observed;
- hidden within-packet uncertainty with factorized slots: FAIL;
- packet-level latent: PROMISING but unresolved;
- CPU low-batch throughput advantage: PASS in this fixture;
- GPU / natural-language speedup: NOT ESTABLISHED.

## Architecture implication

A plausible next stack is:
past context -> packet-level latent/plan -> P parallel phase slots -> optional sparse shared rules -> P tokens.

This separates temporal packet parallelism from functional MoE/shared-rule selection. The next experiment should test whether a learned packet latent or fixed-depth intermediate-state interface can preserve joint consistency without sequential token feedback.
