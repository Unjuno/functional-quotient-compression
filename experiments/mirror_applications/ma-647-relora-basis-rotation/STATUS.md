# MA-647 status

- Status: FAIL
- Branch: `research/ma-647-relora-basis-rotation-20261104`
- Base commit: `3a135653`
- Protocol frozen before development: yes
- Development complete: yes (64701, 64702)
- Fresh/audit opened: no; frozen native-control alias gate reached

## H / T / D / C / U

**H:** One shared butterfly angle schedule plus a per-restart scalar view can generate diverse rank-4 update subspaces with lower payload than independent ReLoRA/BOFT factors while recovering a rank-16 cumulative function.

**T:** Closed-form synthetic 32x32 target assembled from four rank-4 butterfly-rotated updates, with five stages and 16 pair angles/stage. Two development seeds. Compared independent ReLoRA factors, independent BOFT angle tables, shared-angle/scalar Mirror, native shared-angle BOFT, and static unrotated rank-4 basis. Actual `.npz` inference bytes, relative Frobenius error, effective rank, rotation/decode operation proxy and serialization wall time measured. No optimizer updates.

**D:** FAIL for Mirror-specific claim. The shared scalar schedule recovers rank16 with zero measured error and payloads of 7,412B/7,421B, 7.6% below independent ReLoRA and 4.8% below independent BOFT. But the native shared-angle BOFT control has identical payload and exactly equal basis, angle table, scales, coefficients, reconstructed function, quality and compute. Static rank4 misses rank growth (relative error .831/.813). Fresh seeds 64711–64713 stayed sealed.

**C:** The Mirror coordinate is direct scalar modulation of a butterfly angle schedule, which is already a native shared-parameter BOFT method. The apparent byte gain comes from sharing the transform parameters and is reproduced exactly by BOFT.

**U:** No optimization convergence or language model behavior was tested. Whether an address can improve over better native structured-rotation parameterizations remains open outside this tested scalar-schedule design.

## Fact / Interpretation / Hypothesis

- **Fact:** 10 rows (2 seeds x 5 methods); actual serialized bytes and reconstruction metrics replay; three tests pass; protocol hash matches freeze.
- **Interpretation:** A structured shared rotation can preserve logical rank growth and save bytes versus independent transforms, but the gain is ordinary BOFT parameter sharing, not Mirror-specific.
- **Hypothesis:** The ReLoRA restart-coordinate subfamily needs a design that beats direct BOFT/codebook controls before further variants are run.
