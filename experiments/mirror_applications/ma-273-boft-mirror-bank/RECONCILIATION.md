# MA-273 protocol reconciliation

Two dedicated branches registered MA-273 with different task generators, controls, and evidence lanes. Their metrics are not pooled. Root disposition is FAIL because neither establishes a Mirror-specific advantage.

## Protocol A: post-fit aligned orbit and stress audit

**Fact:** `research/ma-273-boft-mirror-bank-screen-20261008` ran corrected fresh seeds 163/269/367/463 after the initial serialization replay defect was detected. The aligned teacher used a shared 12-angle butterfly atom scaled by six scalar task codes. On aligned seeds, independent BOFT used 1,582B, while Mirror and ordinary scalar-times-shared-angle factorization both used 1,150B (27.3% fewer) and both achieved numerical-zero MSE. In independent-angle stress, both factorized models had mean MSE 0.06944 while independent BOFT was numerical zero. Orthogonality and inverse-cycle checks passed at numerical precision.

**Interpretation:** The post-fit orbit has a compact factorization, but the `m` notation adds no function, bytes, or quality beyond the ordinary scalar-factor control. Unrelated butterfly coordinates require more state.

**Hypothesis:** Shared butterfly atoms may remain useful as a generic structured factorization when tasks align to a low-dimensional angle subspace.

## Protocol B: neutral-initialization trained screen

**Fact:** `research/ma-273-boft-mirror-bank-20261008` used a 16D four-task teacher, 1,000 updates, two development worlds, and LR selection. After an initialization audit removed copied target angle/code leakage, the selected LR 0.01 produced Mirror payload 3,593B vs native per-task BOFT 3,787B (about 5.1% fewer), with Mirror task MSE 0.00060–0.00353 vs BOFT 0.00037–0.00170 and independent maps near 1e-9. Fresh seeds 27302–27304 remained sealed. The initially leaked run is separately preserved and excluded.

**Interpretation:** The neutral-init candidate failed the fixed-update development quality criterion; the small byte reduction did not compensate. It does not prove representational impossibility because optimization may not discover the bilinear factorization.

**Hypothesis:** Better initialization or private angle residuals could improve learned fit, but require a separately preregistered experiment and would add paid state.

## Decision and boundaries

**D: FAIL** for Mirror-specific advantage. Protocol A has exact simple-control equivalence; Protocol B fails its development gate and has no fresh evidence. Preserve both scopes independently. No neural language-model, diffusion, general BOFT, near-convergence capacity, or adoption claim is established.

**Strongest counter-hypothesis:** Both tasks are synthetic and deliberately low-dimensional; Protocol A builds the shared factorization into the teacher, while Protocol B may confound representational ability with optimization difficulty.

**Unconfirmed:** Whether learned butterfly Mirror addresses improve downstream task quality over native BOFT or ordinary factorized controls after a newly preregistered training protocol.
