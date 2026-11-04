# MA-663 status — scoped FAIL; SSM table factorization lane paused

- Decision: **FAIL for a Mirror-specific effect.** Shared channel × timescale factors store recurrence modes compactly, and private factors recover residual dynamics, but native CP represents exactly the same functions and paid state.
- Development seeds: 66301/66302. Fresh seeds 66311–66313 remain sealed after exact native CP state/output alias.
- Protocol was amended before fresh access to clarify the private-residual quality tolerance as absolute NRMSE `<= full-model error + 0.02`; the original is retained at `source/attempts/protocol_v1_before_metric_clarification.json`. The status conclusion and selected ranks do not depend on this clarification.
- Conditions: 8 channels × 8 timescales; 16/64 tasks; sequence length 32; 16 sequences/task; shared CP rank 4; off-subspace residual RMS 0/.05/.2; private residual ranks 0/2/4/8.
- Actual payload bytes include transition state, task codes, readout, private residual factors and metadata. Generator and recurrent MAC proxies are reported separately.

## H / T / D / C / U

- **H — Hypothesis:** task coordinates over shared channel and timescale factors can store many useful recurrence tables compactly; private residual rank marks when independent task state is required.
- **T — Trial:** 240 rows from two development seeds across task counts, residuals and private ranks. A stable softplus maps raw transition tables to rates in a diagonal recurrence. Compared independent full tables, shared rank-4 CP, and the same CP state under Mirror and native labels, with private residual ranks. Evaluated sequence-output NRMSE, actual serialized payload bytes, generation and recurrent compute proxies.
- **D — Decision:** FAIL for Mirror attribution. At residual 0, rank-4 shared CP has zero output error and uses 2,131 B (N=16) / 2,940 B (N=64) versus 4,723 / 15,196 B full. At N=64, rho=.2, private rank2 restores mean NRMSE to .0177 at 11,026 B (72.5% of full); N=16 needs rank4 for .0128 error but then costs 6,452 B versus 4,723 B full. Rank8 exceeds full bytes. Mirror/native CP and Mirror/native CP+private arrays and outputs are exact aliases. Fresh stayed sealed.
- **C — Strongest counter-hypothesis:** ordinary CP tensor factorization plus per-task coefficient vectors fully explains the compression and residual frontier; no additional Mirror-specific operation is needed. The synthetic aligned teacher directly generates rates from the same four CP factors.
- **U — Unconfirmed:** trained SSM transitions, language-model quality, competitive retrieval, long-horizon stability, actual GPU generation latency, and whether a distinct structured code can beat CP/Tucker controls on natural sequence tasks.

## Fact / Interpretation / Hypothesis

- **Fact:** tests passed; all 240 rows match actual `.npz` file sizes; metrics replay; Mirror aliases native CP exactly for shared and private residual variants. At rho=0 the shared CP state reconstructs all sequences to numerical precision. At N=64, rho=.2, rank2 private residual has mean NRMSE .0177 and 11,026 B versus 15,214 B full.
- **Interpretation:** channel × timescale factors can compress aligned recurrence tables by 55–81%; at larger task count, modest private residuals can retain a useful storage frontier. This is a tensor-factorization result, not Mirror-specific evidence.
- **Hypothesis:** any later SSM proposal should add a mechanism beyond static channel/timescale basis codes—such as input-dependent retrieval or stability-aware allocation—and compare it against matched native tensor and residual controls.

## Queue ruling

MA-661 and MA-663 are consecutive failures because standard shared low-rank/CP factorization exactly reproduces the Mirror representation. Pause MA-664's shared-transition-basis plus sparse-private-residual variant pending a native-distinct design. MA-665's Hyena filter generator is a different implicit-operator mechanism and remains separately testable.
