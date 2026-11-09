# MA-738 — MoE neural-operator bank compressed by Mirror views

Status: FAIL (scoped synthetic Mirror-specific claim)

## Fact

The independent top-1 bank used 1,626 B and matched its generated operators exactly. The universal operator used 559 B but had total/high-band solution nRMSE `.0368/1.097`. Rank-4 shared PCA/Mirror top-1 reached near-zero error but cost 1,976 B, 21.5% more than independent, with 1.6% more active multiplies due to view reconstruction. Mirror and native PCA have identical payload hash, quality and compute. Top-2 roughly doubled active multiplies and yielded high-band nRMSE `.170`; the full expert bank remained smaller than the shared View bank. See `RESULTS_CORE.csv`.

## Interpretation

The designed four-band family is exactly representable with the shared basis, but actual compressed serialization favors the independent table because its smooth spectra compress efficiently. Paying router/coordinate/basis state removed any storage advantage. Top-2 blending of a nearby regime harmed high-frequency quality while increasing work. Mirror adds no value over ordinary PCA expert coefficients.

## Hypothesis result

**FAIL** for Mirror-specific value and storage/compute Pareto. No fresh run was opened after exact native alias and the byte gate miss.

## Frozen protocol / scope

PA186 FNO and PA02 path-constrained/shared MoE routing. Eight synthetic Fourier-diagonal regime operators, 32 modes, rank-4 PCA basis fit on six identities, two held-out identities, 256 smooth forcing spectra/regime. Oracle route metadata and indices are paid. Top-2 mixes 90% own expert with 10% nearest-code expert. The held-out coordinates are projected from known synthetic spectra; this is an oracle mechanism screen, not trained FNO or learned routing. Complete NPZ payload bytes and active spectral multiply proxies are recorded. CPU timings are unoptimized operator multiplication only.

No PDE experiment, trained neural operator, learned router, natural regimes, resolution transfer or production serving claim.

## H / T / D / C / U

- **H:** Shared spectral View experts reduce a routed MoE operator bank's bytes while preserving quality; top-2 is worthwhile if quality gain justifies extra active compute.
- **T:** Seed 73801, 8 regimes, 32 modes, six fit/two held-out, rank-4, independent/universal/native PCA/Mirror, top-1/top-2 controls.
- **D:** **FAIL.** Near-exact top-1 Views cost 1,976 B vs 1,626 B independent and exactly alias native PCA. Top-2 doubles active work and has `.170` high-band error.
- **C:** Actual NPZ compresses the smooth independent spectrum table efficiently; ordinary PCA has the same View function. Oracle labels also avoid router generalization costs.
- **U:** Learned routing, trained FNO, physical PDEs, real regime shifts, true resolution transfer and fused serving kernels.
