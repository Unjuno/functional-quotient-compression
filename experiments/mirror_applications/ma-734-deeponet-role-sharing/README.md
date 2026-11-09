# MA-734 — DeepONet branch/trunk Mirror role sharing

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

With 8 synthetic domains and rank-4 factors, independent branch/trunk maps reached factor/operator nRMSE `2.60e-8` / `5.23e-8` at 1,358 serialized bytes. Hard tying used 750 B but errors were `.421` / `.505`. Output FiLM was worse (`.386` / `.505`) at 1,653 B. Mirror role Views reached the same near-zero errors as untied at 1,552 B. The native direct-Givens control matched Mirror in complete payload SHA-256, bytes, metrics and active multiply count. See `RESULTS_CORE.csv`.

## Interpretation

The constructed teacher has a shared nonlinear latent map and role-specific 4D rotations. Role coordinates restore the branch/trunk asymmetry that hard tying loses. In this setup, however, the direct native Givens code is the exact same representation, and compressing the duplicated weights in the serialized untied bank makes that bank smaller than Mirror. FiLM cannot express the rotated role geometry. This is a negative Mirror-specific result and a narrow aligned-feasibility example.

## Hypothesis result

**FAIL.** Mirror fails its >=25% storage saving versus the untied serialized control and exactly aliases native direct Givens. No fresh world was opened. Nothing here establishes a trained DeepONet or PDE result.

## Frozen protocol / scope

PA187. Each synthetic operator output is the inner product of branch and trunk factors, each produced by `tanh(Wx)` followed by a role-specific 4D Givens rotation. Eight domains; input width 8; rank 4; 512 training examples/role/domain used only to fit the output FiLM control; 256 held-out examples/role/domain; 4,096 branch-trunk pairs/domain. The shared map and role angles are an analytic teacher, not learned model state. The runner measures held-out factor and composed operator nRMSE, complete compressed NPZ bytes, multiply proxy and CPU timing. No optimizer updates.

Gates: both errors <=.02, >=25% savings vs untied and strict quality/byte improvement over native Givens and FiLM. Exact native alias is FAIL. No physical PDE, trained DeepONet, sensor generalization or hardware latency claim.

## H / T / D / C / U

- **H:** A shared latent map with compact role Views preserves distinct branch/trunk functions while saving bytes and improving over native controls.
- **T:** Seed 73401, 8 domains, rank 4, input width 8; untied, hard tied, output FiLM, direct Givens and Mirror; fixed held-out feature inputs. Development mechanism screen only; no fresh run.
- **D:** **FAIL** for Mirror-specific value. Mirror uses 1,552 B vs 1,358 B untied and exactly aliases native direct Givens; the low-error role-view mechanism itself works on the aligned teacher.
- **C:** Role asymmetry is ordinary Givens conditioning, and lossless NPZ compression captures duplicated shared weights in the untied payload.
- **U:** Learned DeepONet training, natural branch/trunk distributions, sensor-layout variation, PDE transfer and production runtime.
