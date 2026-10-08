# MA-009 — shared Mirror experts plus rare private expert

Status: **FAIL** under the pre-fresh quality gate. The mechanism strongly improved over shared-only controls, but rare-private Mirror missed the full-MoE relative quality threshold in two of three aligned worlds.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-009-rare-private-mirror-expert-20261007`.
Base commit: `f45deaeccb` (full SHA in `PROTOCOL.json`).

## H — Hypothesis

A private matrix for a rare outlier role plus Mirror views for common aligned roles recovers full-MoE quality at lower serialized payload; all-shared Mirror should fail at the outlier.

## T — Test

A 16D-to-12D hard-routed synthetic MoE. Positive shifts on the first two input coordinates make role 0 occur about 1.3% of the time. Compared full independent MoE, all-Mirror, hard tying plus rare-private, rank-1 residuals, and Mirror common views plus a private rare matrix. Teacher modes were shared-base Givens common roles plus an independent rare expert, and four independent matrices. Development world 90000 selected LR 0.01. Fresh worlds 90001–90003 used matched data and 1,200 updates. Gate amendment from 0.70x to 0.80x full payload was made after dev but before fresh; it still required at least 20% fewer actual serialized bytes.

## D — Decision: FAIL (formal quality gate)

### Fact

- In the aligned common/private-rare condition, natural-frequency Mirror MSE was 3.94e-6 / 4.71e-6 / 7.48e-6 vs full MoE 2.65e-6 / 7.24e-6 / 5.31e-6. Relative ratios: 1.49x / 0.65x / 1.41x. Rare-role MSE ratios were likewise 1.49x / 0.65x / 1.41x. This misses the <=1.10x quality gate in two worlds, so it meets the preregistered FAIL rule.
- The amended actual-payload gate passed in all worlds: 3,745B vs 4,841B (0.774x). Mirror-private substantially beat all-Mirror (natural MSE ~0.10, rare-role MSE ~7–8) and tied/rank-1 common controls (natural MSE ~0.18–0.34).
- For independent matrices, Mirror-plus-private-rare MSE was 1.52–1.73 vs full MoE 2.7e-6–1.7e-5. One private expert is not enough for arbitrary expert functions. Rank-1 residuals were better than Mirror-private in all three independent worlds (1.35–1.44), at 3,361B vs 3,745B.
- Mirror active-compute proxy was 15.97M vs 14.75M full MoE (+8.3%). Median aligned training time was 7.44s vs 2.07s; median CPU inference throughput was 64k vs 1.05M examples/s (0.061x).
- 30 fresh rows replayed; payload bytes matched exactly, maximum natural MSE delta 4.7e-10, rare-role delta 5.0e-10, balanced MSE delta 4.4e-10. Tests: 4 passed.

### Interpretation

Allocating one private role substantially improved over all-shared parameterizations and saved 22.6% payload, but it did not meet the strict full-MoE rare-quality gate reliably. Under this natural frequency, one private matrix repaired most of the rare function, yet two fresh initializations left a small absolute quality gap. The eager Givens path has a large runtime penalty. This candidate does not establish a usable Pareto improvement under its preregistered gate.

### Strongest counter-hypothesis

The failed relative threshold may reflect finite-update and initialization variation on a very rare role: all errors are small in absolute terms and full MoE is also fitting near the numerical floor. The data do not isolate an intrinsic capacity limit for one private matrix.

### U — Unconfirmed

Longer training, rare-role oversampling, larger nonlinear experts, a learned router, optimized fused views, and capacity near convergence remain untested. Do not convert this failure into a general claim that private experts are always required for rare roles.

## Fact / interpretation / hypothesis

- **Fact:** storage passed; relative quality failed 2/3; independent-all required more capacity; current CPU runtime regressed sharply; replay and tests passed.
- **Interpretation:** the tested private allocation is promising against shared-only controls but failed its full-MoE quality requirement in this fixed-budget setup.
- **Hypothesis:** oversampling or more updates could stabilize the rare private expert; this needs a new preregistered experiment and cannot be inferred from these fresh worlds.
