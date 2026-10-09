# MA-405 status

Status: **FAIL** at the frozen development gate.

## H
Non-diagonal Givens views may complement StyleGAN2 channel modulation on rotation-dominated FFN context variation.

## T
Shared 32D FFN, eight contexts, five mixtures, five controls; seeds 40501 and 40502; 4096 train plus separate 4096 held-out inputs/task; 400 updates. Fresh stayed sealed.

## D
FAIL. At alpha 0, Mirror NRMSE was 0.000447/0.000473 against StyleGAN2 0.636/0.548. Payload was 0.874–0.876x StyleGAN2 and 0.0992–0.0993x independent full. Seed 40501 throughput was 0.633x StyleGAN2, below the 0.80 threshold; seed 40502 reached 0.996x. The two-seed screen therefore failed.

At alpha 1, StyleGAN2 was accurate and Mirror was not. For mixed targets, both structured controls lost accuracy while independent matrices remained below 0.007 NRMSE.

## C
The endpoints are deliberately aligned with the tested mechanisms; results do not establish natural-task utility. The runtime gate missed in one seed, and the byte gain over StyleGAN2 was modest.

## U
Natural task quality, fresh replication, optimized kernels, and whether composed style+Mirror codes improve the frontier.

FACT: all 50 metrics replay from serialized inference payloads. INTERPRETATION: two complementary endpoint families are compact; their mixtures need private matrix state. HYPOTHESIS: composition may help but requires a new protocol.
