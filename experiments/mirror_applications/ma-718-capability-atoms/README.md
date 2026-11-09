# MA-718 — merge-many then factor reusable capability atoms

Status: SCREENING

## Hypothesis

H: A shared capability atom basis fit to merged source task deltas can recover individual source capabilities and their combinations at lower complete serialized bytes than sparse DARE/DELLA controls, while beating ordinary PCA/shared-basis coding.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, a sparse coefficient vector over shared task-delta capability atoms, so that multiple individual and composed logical capabilities can be retained without storing every dense source delta.

## Prior art and controls

PA175 DARE; PA176 DELLA; PA27 TIES. Controls: independent dense delta bank, random DARE top-10%, magnitude top-10% (DELLA-inspired), ordinary PCA rank-coded bank, Mirror atom code, and exact native PCA/shared-dictionary alias control. Task Arithmetic and TIES composition are measured on requested pairs as interference references.

## Frozen development protocol

Synthetic bank: D=512, 24 source deltas, 16 basis-fit tasks and 8 held-out whole tasks, latent shared capability rank 6 plus controlled private Gaussian component (rho=.02), dev seed 71801. Fit shared basis on the first 16 tasks only. Sweep basis ranks 2/4/6/8/12/16 and retain coefficients in float32. Compare held-out individual reconstruction NRMSE and four pairwise composed deltas against the same target sum. Serialize complete bank-level state once, charging basis, codes, indices, values, private residuals, request IDs, and all source/output state. Development-only; no audit task identities are accessed.

### Gates

PASS requires held-out individual and composed NRMSE <=.05, >=25% complete bank byte savings versus independent and both sparse controls, and improvement over native PCA at matched bytes. Native PCA exact function/payload alias is FAIL for Mirror-specific value. If all quality gates fail on the first frozen setting, stop this screen without opening audit.

## Scope

This is a synthetic linear delta mechanism test. It cannot establish natural model merging, task utility, language quality, or production runtime.

## Results — development seed 71801

**FACT:** Independent full bank plus four pair outputs is 53,960 B. Native PCA rank-6 bank is 20,379 B (62.2% fewer bytes) with held-out task NRMSE .009845 and pair-composition NRMSE .009831. Rank-4 is smaller (16,452 B) but held-out NRMSE .503. Random top-10% DARE-inspired and magnitude top-10% DELLA-inspired controls are 8,916 B/.949 and 8,605 B/.752 held-out NRMSE. At rank 6, Mirror atoms and native PCA have identical payload hash, bytes, reconstruction and composition. Active reconstruction work proxy is 73,728 vs 12,288 for dense (6x); no hardware runtime claim.

**INTERPRETATION:** A shared low-rank atom bank preserves the deliberately aligned synthetic capability family with useful compression, but ordinary PCA/shared-basis coding fully explains the result. Sparse pruning alone is smaller and loses function quality. Rank-6 decode has a higher operation proxy; no Pareto improvement attributable to Mirror is established.

**HYPOTHESIS:** The generated shared rank-6 structure is favorable by construction; unrelated natural task deltas may require private residuals or fail to compress. No natural result follows.

**D:** FAIL for Mirror-specific value. The method can represent this aligned synthetic family, but native PCA exactly aliases it. Fresh/audit unopened.

**C:** The task bank was generated from a shared six-atom latent structure, which strongly favors PCA and says little about learned model deltas. DARE/DELLA/TIES were only simplified controls; Task Arithmetic and TIES were not separately serialized as control rows in this screen, a deviation disclosed here.

**U:** Real pretrained model deltas, independently trained tasks, model quality/retention, exact DARE/DELLA/TIES reproductions, private-residual frontier, fresh seeds and production runtime.

### H / T / D / C / U

- **H:** Shared capability atoms recover source tasks and compositions with lower complete bytes than sparse/native alternatives and add value beyond PCA.
- **T:** D=512, 24 tasks (16 basis fit, 8 held out), latent rank 6, seed 71801; independent, DARE-inspired, DELLA-inspired, PCA rank sweep, Mirror atom and four held-out pair compositions.
- **D:** FAIL for Mirror-specific attribution; rank-6 PCA gives NRMSE .0098 at 20,379 B, but Mirror is exact payload/function alias and projection work is 6x dense proxy.
- **C:** Favorable synthetic aligned latent construction; simplified sparse baselines.
- **U:** Natural-model and audit evidence.
