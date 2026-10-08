# MA-492 — Discrete packet-plan Mirror latent

## H — hypothesis

A compact discrete packet-plan code can condition one shared packet decoder and improve joint future-mode quality at lower serialized state than independent per-mode decoders. The code must add value over the least costly unconditional model and same-bit native latent controls.

## T — execution

Frozen protocol uses a synthetic four-step binary packet represented by one of 16 categories. Each world samples four valid packet modes and a mixture; 4096 support and 4096 fresh heldout packets. Development seed 49201; fresh seeds 49211–49213. Compared an unconditional shared categorical decoder, 2-bit discrete Mirror addresses, byte-matched VQ, continuous latent decoder, and four independent full 16-way mode decoders. No gradient updates; each fit uses 4096 examples. Actual canonical NPZ payloads were serialized and counted. Runtime is tiny CPU fit calibration only.

## D — FAIL

Across all three fresh worlds Mirror and VQ had exactly identical joint NLL, valid mass, coverage and 817-byte payload, by construction of their identical inference state. Mirror NLL equaled the cheaper unconditional model to the reported precision (1.3571, 1.2982, 1.1716 nats/world); unconditional payload was 603 B. Mirror did not meet the required >=0.05 nats improvement and cost 214 B more than unconditional. Independent full-mode decoders gave slightly lower NLL (by about 0.00073 nats) at 822 B, not a meaningful Mirror advantage. Continuous latent used 1023 B and did not improve quality. The frozen quality/specificity gate failed in 3/3 worlds.

## C — strongest counter-hypothesis

This synthetic categorical packet problem is already solved by an unconditional empirical distribution. The discrete Mirror is ordinary VQ-style conditioning with no marginal functional value; apparent 2-bit mode multiplicity is only address count, not measured added capacity.

## U — unconfirmed

No language-model packet generation, PTP comparison, long packet horizon, neural decoder training, or real sequence distribution was tested. This is a small categorical mechanism screen, not a PTP result.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror and same-bit VQ are exactly equal in payload and predictions in every fresh world. Mirror 817 B vs unconditional 603 B; independent full bank 822 B; continuous latent 1023 B.
- **Interpretation:** The registered code did not create useful joint-mode quality beyond the shared empirical distribution. It adds storage without utility in this task.
- **Hypothesis:** A packet plan may help when context contains predictable conditional mode information that an unconditional distribution cannot represent; that remains untested.

See `RESULTS_CORE.csv`, `source/fresh_summary.json`, and the frozen `PROTOCOL.json`.
