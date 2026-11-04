# MA-670 status — scoped FAIL for Mirror attribution

- Decision: **FAIL for a Mirror-specific gain.** A shared rank-1 source basis compresses aligned LoRA banks and precomposed targets substantially; ordinary VeRA-style shared factors and scaling vectors reproduce the same state/functions and use 44 fewer serialized bytes due to metadata field names.
- Protocol frozen before development: commit `a9cda1a5`; hash is in `freeze.json`.
- Development seeds: 67001/67002. Fresh seeds 67011–67013 remain sealed after the exact VeRA/shared-scaling alias.
- Conditions: dimension 16; 16/64 source LoRA modules; 16 target compositions; shared atom rank4; private source residual rank2; residual RMS 0/.05/.2; 128 held-out queries per target.
- Target mixture coefficients are known synthetic task metadata. The experiment isolates source-bank compression; search is MA-669's question.

## H / T / D / C / U

- **H — Hypothesis:** Mirror source codes can compress a LoRA library before target composition, preserving composed outputs at lower total bytes and composition compute than LoRAHub's complete source factors plus coefficients, beyond VeRA scaling.
- **T — Trial:** two development seeds compared full rank≤6 source LoRA factors plus dense target coefficients, independent target matrices, rank-1 shared atoms plus source/target Mirror codes, and identical native VeRA-style scaling. Private rank-2 residual factors and target coefficients are included for nonzero residuals. Measured held-out composed-output NRMSE, actual serialized bytes, composition MAC proxy and query inference MAC proxy. Fresh remained locked.
- **D — Decision:** FAIL for Mirror attribution. At rho=0, N=64 source modules plus 16 target mixtures used 5,704 B and 20,480 composition MACs for Mirror versus 53,613 B and 262,144 MACs for LoRAHub; outputs were exact. At N=16, Mirror used 4,893 B versus 15,371 B. However, native VeRA/shared-scaling reproduced the same arrays and functions and used 44 B fewer. At rho=.05/.2, private rank2 retained exact output quality but N=64 total state grew to ~40.7 KB and composition compute to 544,768 MACs; N=16 grew to ~13.8 KB. This identifies the private-state cost boundary. Fresh stayed sealed.
- **C — Strongest counter-hypothesis:** the result is VeRA-style factor sharing: shared left/right rank-1 matrices and per-source scale vectors exactly define the source LoRA bank and all composed targets. No additional Mirror coordinate is required.
- **U — Unconfirmed:** real LoRA banks trained on language tasks, approximate basis learning, noisy target compositions, target coefficient search, GPU fusion/latency, and generalization to new source adapters. This is a synthetic function-family screen.

## Fact / Interpretation / Hypothesis

- **Fact:** 48 rows replay; all serialized file sizes match recorded bytes; two tests pass. Mirror and native VeRA physical factors, source/target codes, private residuals and outputs are identical. At N=64/rho=0, Mirror's 5.7 KB is 10.6% of the LoRAHub source-bank payload.
- **Interpretation:** source-bank compression and precomposition produce a strong storage and composition-compute point when all source adapters share a rank-4 atom basis. Private residuals retain quality but consume most savings and can exceed independent target storage.
- **Hypothesis:** a useful Mirror result needs a source family where structured View coordinates improve quality or total deployed state beyond VeRA, native factor sharing, and explicit adapter dictionaries.
