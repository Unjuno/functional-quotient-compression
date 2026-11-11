# MA-852 — FAIL for Mirror-specific attribution

## H / T / D / C / U

- **H:** Task-specific decay coordinates on one shared delta-write rule retain context-dependent sequence memory with lower storage than independent full per-task update matrices.
- **T:** Synthetic D=16, 8 tasks, 64-token streams, 16 one-hot keys; score 16- and 64-token prefixes. Development seed 23 selected fixed λ=.45; fresh seeds 43 and 71. Controls: fixed λ, native task scalar gate, Mirror code, and redundant full λI operator per task. 512 support tokens and updates per world; no optimizer training.
- **D:** **FAIL** for Mirror-specific attribution. Native gate and Mirror both have zero output nRMSE, 8,788 B payload, and identical 262,144 update + 65,536 query MACs; Mirror does not add a distinct function. Full per-task operator control also has zero error but costs 16,946 B. Fixed λ is smaller (8,762 B) but has fresh nRMSE .346 and .263. State reset residual is 0.
- **C:** A task-indexed scalar decay gate is a standard native parameterization that exactly explains the retention and compression result.
- **U:** Learned context inference, noisy/repeated key tasks beyond this synthetic family, trained DeltaNet/Mamba, long sequence throughput, and model-scale language quality.

## Facts

Each task supplies its scalar retention context λ. For each token, the tested rule updates W←λW+vkᵀ. Rates and states are charged in the serialized payload. The full-operator bank stores one λI matrix per task; gate/Mirror store one scalar/task. Payloads were reloaded and compared exactly. One writable W is reused and cleared in place between task episodes.

## Interpretation

Task-conditioned forgetting recovers retention behavior that a single fixed rate cannot express, but a native scalar gate matches the Mirror coordinate exactly. The parameter bytes saved relative to a deliberately redundant full-operator bank are due to scalar parameterization. This does not establish Mirror-specific capacity or a trained DeltaNet result.

Each active matrix state is 1,024 B; the 8-task writable state bank is 8,192 B and is included in every serialized payload. Thus the code/operator differences compare rule storage on top of equally charged writable state.
