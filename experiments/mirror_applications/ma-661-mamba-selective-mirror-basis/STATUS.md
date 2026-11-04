# MA-661 status — scoped FAIL

- Decision: **FAIL for the frozen Mirror storage gate and Mirror-specific attribution.** Shared input bases yield useful compute reductions on the aligned task, but are ordinary native basis factorization.
- Protocol frozen before development: `08576061`; hash is in `freeze.json`.
- Development seeds: 66101/66102. Fresh seeds 66111–66113 remain sealed after the development gate miss and exact native joint-basis alias.
- Conditions: 8D input/state, length 32, 128 sequences per world; shared ranks 1/2/4/8; off-basis residual RMS 0/.1/.3.
- Methods: full input-selective diagonal SSM; independent rank-k B/C/Delta projections; joint shared-input-basis B/C/Delta; same shared basis implemented as a Mirror coordinate.
- Complete actual `.npz` inference bytes include dynamics, B/C/Delta generator state, basis/factors, biases and metadata. Compute reports generator MAC proxies and recurrent update proxies.

## H / T / D / C / U

- **H — Hypothesis:** a per-token Mirror code `m(x)=Vx` can factor input-dependent B/C/Delta generators and improve the sequence-quality/storage/compute frontier beyond full native projections and matched native low-rank methods.
- **T — Trial:** synthetic selective diagonal SSM on two development seeds, 128 held-out sequences per seed/residual, with ranks 1/2/4/8 and residual RMS 0/.1/.3. Evaluated output NRMSE, actual serialized payload bytes and generator MAC proxy. Included full Mamba-like projections, independent low-rank factors, and joint native basis controls. Fresh remained locked.
- **D — Decision:** FAIL. At rho=0, rank4 reconstructs sequence output at numerical precision, but mean payload is 1,932 B versus 2,004 B full native (only 3.6% smaller; frozen gate required at most 60%). The native joint-basis control is exactly the same arrays and outputs. At rho=.1/.3, rank4 mean NRMSE is .308/.568; rank8 restores zero error but uses ~2,472 B and 1.05M generator MACs versus ~2,004 B and .79M for full native. Rank4 does lower generator MACs to .52M on the aligned case, but this is reproduced exactly by native joint basis.
- **C — Strongest counter-hypothesis:** common input-subspace factorization is a standard joint low-rank projection. It explains the coordinate code, bytes, quality, and generator compute without an additional Mirror capability. The task also intentionally favors a rank-4 shared input subspace at rho=0.
- **U — Unconfirmed:** trained Mamba, language-model perplexity, long-sequence stability, GPU latency, and whether task-trained projections or nonlinear selective generators have a stronger frontier. The measured compute is a MAC proxy, not hardware runtime.

## Fact / Interpretation / Hypothesis

- **Fact:** 96 development rows cover all seeds/ranks/residuals/methods. Two tests pass, replay all development metrics, match actual payload file sizes, and verify exact Mirror/native joint-basis arrays.
- **Interpretation:** a shared input basis can reduce token generator MACs by one third for rank4, but at this small dimension its serialized payload barely improves on full native, and the shared-basis result is ordinary low-rank factorization.
- **Hypothesis:** higher-dimensional trained selective SSM generators could amortize the shared basis enough for a meaningful storage gain; a Mirror claim requires a result beyond native joint projection and low-rank controls.
