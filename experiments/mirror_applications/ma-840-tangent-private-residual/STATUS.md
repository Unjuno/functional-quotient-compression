# MA-840 status

Status: **FAIL** for Mirror-specific advantage; scoped synthetic screen. Fresh data remained sealed after native PCA/residual aliases and gate misses.

## H / T / D / C / U

- **H:** A small support-derived private residual over a shared tangent basis would recover nonlinear exceptions at >=25% fewer complete bytes than independent updates and beat native factorization/VeRA controls.
- **T:** Synthetic 6-12-1 tanh MLP, 10 tasks, six ordinary tasks for the shared basis and four heldout tasks including two private-direction exceptions; support/test 256; scales .03/.20; ranks 0/1/2/4; no optimizer updates. Focused test: 1 passed. The private residual basis is fitted from support-derived residuals, including heldout-task support; no unseen-exception transfer is tested.
- **D:** At scale .03, shared-only rank0 reaches heldout nRMSE .01405 (exception tasks .01916) at 3,091 B, 33.9% fewer bytes than independent tangent deltas (4,680 B). Rank4 lowers error to .00312 but costs 4,783 B, above independent; ranks1/2 are worse (.03590/.03932). At scale .20, tangent residual is .04208 and no Mirror/native rank meets the .02 gate; rank4 error .5683 at 4,796 B. VeRA random scaling errors are .0250 and .1664. Every Mirror/private-rank payload exactly aliases its native PCA/residual payload. Fresh remained sealed.
- **C:** For small perturbations, the shared basis alone already meets the loose .02 threshold; extra private rank is not needed for that gate. Improving error from .014 to .003 requires rank4 and erases storage savings. At larger perturbations, local tangent error dominates.
- **U:** No pretrained or language model, actual optimizer training, natural task exceptions, fresh seeds, or runtime on production hardware. The two exception directions and linearization scale are synthetic.

## Fact / Interpretation / Hypothesis

- **Fact:** Native PCA/residual and Mirror payload bytes, hashes and errors match at all ranks. At .03, rank0 meets the chosen quality and byte gates; rank4 improves error but exceeds independent bytes. At .20 all tangent variants miss quality.
- **Interpretation:** The storage/quality frontier is controlled by basis rank and linearization validity; Mirror adds no function beyond native factorization. A private component can improve quality, but the rank required here removes the byte advantage.
- **Hypothesis:** Shared tangent state can cover near-linear tasks, while private nonlinear capacity becomes expensive when tighter quality is required; beyond the local linear regime, residual rank alone may be insufficient.
