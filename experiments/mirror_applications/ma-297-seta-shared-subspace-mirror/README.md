# MA-297 — SETA shared sparse subspace + Mirror views

Status: SCREENING. Base: `2f4d01ba`. Prior art: PA30 SETA.

## H / hypothesis

After discovering shared sparse coordinates from earlier tasks, a compact task-coordinate basis inside that region reduces storage growth while preserving later task functions. Generic PCA may explain any gain.

## T / protocol

Synthetic sparse linear tasks (D=128, eight tasks, 16 shared coordinates and four private coordinates per task). Discover shared support using the first six tasks, then compare independent sparse vectors, shared support with private residuals, Mirror task codes, and generic PCA. The encoder receives task deltas directly; this tests reconstruction/storage, not learning. Development worlds 29700–29701; fresh worlds 29710–29712; three seeds each. Index arrays, values, basis, codes and metadata are included in serialized payloads.

This is a SETA-inspired storage screen, not a faithful continual-learning reproduction. It does not test routing, sequential updates or forgetting.

## Development screen

Shared support was recovered in all seeds. SETA-style shared/private uses about 1,055 B; Mirror task codes use about 836 B and generic PCA about 841 B, both at query NRMSE about 3.6e-7. This meets the storage screen versus the chosen SETA representation, while Mirror-specific value is unsupported because generic PCA matches it. Fresh results remain unopened.

## C / strongest counter-hypothesis

Low-rank PCA over discovered shared coefficients provides the same compression and quality as the Mirror code.

## U / limits

Synthetic linear tasks only; no language-model or continual-retention claim.
