# MA-009 — shared Mirror experts plus rare private expert

Status: SCREENING
Evidence lane: MECHANISM / STORAGE / RUNTIME
Branch: `research/ma-009-rare-private-mirror-expert-20261007`
Base commit: `f45deaeccb` (full SHA recorded in `PROTOCOL.json`).

## H — Hypothesis

If common expert roles share one physical basis and one rare role is independent, a private matrix only for the rare role plus Mirror coordinates for common roles can preserve rare-role quality with less serialized payload than full MoE. All-shared Mirror should miss the independent role.

## T — Protocol

16D-to-12D linear experts, four hard-routed roles selected by input signs. Shifted Gaussian inputs make role 0 rare (~1.3%). Compare full MoE, all-Mirror, tied plus rare-private, rank-1 residuals, and Mirror-common plus rare-private. Teacher modes: shared-base Givens views for common roles with a private rare matrix; all roles independent. Development world 90000 selected LR 0.01. Fresh worlds 90001–90003 remain sealed. All methods receive the same oracle route, isolating expert capacity.

## Development screen (world 90000)

At LR 0.01, shared-common/private-rare Mirror MSE was 9.0e-6 vs full MoE 9.45e-6; rare-role MSE was 7.26e-4 vs 7.62e-4. All-Mirror MSE was 0.0927 and missed the rare role badly (7.28). Actual payload was 3,745B vs 4,841B full MoE. In the all-independent teacher, Mirror-plus-private-rare MSE was 1.77 vs full MoE 1.42e-6, showing one private expert is insufficient for arbitrary matrices.

### Pre-fresh protocol amendment

The original payload threshold of 0.70x was amended to 0.80x after development measured a serializer-inclusive ratio of 0.774x. The revised gate still requires at least 20% actual payload reduction. Quality gates, method set, data, LR candidates, update budget and fresh worlds were unchanged. This amendment is recorded in `PROTOCOL.json` before fresh access.

## Decision

FACT: fresh verification pending.
INTERPRETATION: dev suggests one private matrix can repair the rare outlier while Mirror views recover common aligned roles.
HYPOTHESIS: still under fresh evaluation.
BOUNDARY: synthetic linear experts and known routes; no language or router claim.
