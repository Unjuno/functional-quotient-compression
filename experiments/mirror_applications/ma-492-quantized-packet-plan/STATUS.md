# MA-492 status

- Status: FAIL for registered K<=8 >=95% joint-mode-coverage gate; K16 scoped positive result
- Branch: `research/ma-492-quantized-packet-plan-20261009`
- Base: `7ff75963`; verification commit `3bdd02d2`; A1 amendment before authoritative fresh evaluation
- Fresh: 49220-49222 × seeds 0-2

H: Discrete shared packet-plan codes preserve joint modes with fewer actual bytes than independent marginals.

T: Synthetic 256-context two-step task, 16 legal joint modes; VQ K2/4/8/16 vs continuous, independent categorical, greedy.

D: FAIL strict gate. K8 reached 50% coverage (NLL 10.662); K16 reached 100%, NLL .693, valid packet rate 100% at 3,173B vs continuous 18,469B and independent 18,213B. Independent valid packet rate 59.5%.

C: Single codebook K<=8 cannot cover 16 equally represented legal modes under this protocol.

U: Natural packet prediction, learned plan codes and real decoder latency.

A0 used an invalid mode/coverage construction and is preserved as exploratory only. A1 uses 16 unique legal token pairs and corrected joint-mode coverage/NLL calculations.
