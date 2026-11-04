# Selective SSM shared-factor diagnostic — 2026-11-04

## Decision

Pause MA-664, whose present proposal is a shared SSM transition basis plus sparse private residuals. MA-661 and MA-663 are consecutive P0 failures with the same attribution cause: a native shared low-rank/CP factorization exactly represents the Mirror state and function. Continue MA-665 because its Hyena implicit filter generator is a different operator and must be evaluated on its own merits.

## Evidence

- **MA-661 — input-selective B/C/Delta generators.** At rank 4 and aligned synthetic tasks, output error was numerical zero and generator MACs fell by one third. Actual payload was only 3.6% smaller than full native; native joint-input basis exactly matched Mirror. Off-basis residuals made rank 4 inaccurate; rank 8 cost more bytes and MACs than full native.
- **MA-663 — channel × timescale transition tables.** Rank-4 CP exactly retained aligned functions at 45% of full bytes for 16 tasks and 19% for 64. At 64 tasks with residual RMS .2, private rank 2 restored output NRMSE .0177 at 72.5% of full bytes. Mirror and native CP plus private residuals were exact state/output aliases. At 16 tasks, the needed rank 4 residual exceeded full bytes.

Both are synthetic, use two development seeds, measure actual serialized payloads, and keep fresh seeds sealed after exact native-control alias. They are mechanism screens, not Mamba training or language-model evidence.

## Interpretation

Factorized SSM state can create a real storage/compute frontier when transition or generator functions share a low-dimensional structure. These results attribute the gain to established input low-rank projection and tensor CP codes. The task-specific residual frontier is useful engineering evidence, but it does not establish a distinct Mirror degree of freedom.

## Scope

This pause applies to static shared SSM table/basis coding and residual sweeps represented by MA-664's current proposal. It does not conclude that selective state spaces or implicit operators cannot benefit from a dynamic Mirror coordinate. MA-665's Hyena filter-generator test is structurally separate.
