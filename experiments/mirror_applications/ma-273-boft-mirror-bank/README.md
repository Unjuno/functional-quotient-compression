# MA-273 — BOFT butterfly bank with Mirror task addresses

Status: SCREENING. Evidence lane: MECHANISM / STORAGE / RUNTIME. Base `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: A shared butterfly-orthogonal angle vector plus scalar per-task address can represent task adapters on one BOFT orbit with fewer bytes than independent BOFT transforms; arbitrary factor-angle banks should require more state. The ordinary scalar-times-shared-angle control may be mathematically identical and is mandatory.

> **Mirror insertion:** this experiment adds task coordinate `m_t` to a shared vector of BOFT butterfly angles `φ`, producing logical task transforms Q(m_t φ) without storing each independent BOFT parameter vector.

PA21 establishes butterfly orthogonal transforms. Compare independent BOFT parameter vectors, a shared factorized parameter control with identical function, one shared BOFT, and dense independent maps. No novelty claim for BOFT or orthogonality.

## T

8D frozen linear block, 3 butterfly stages (12 Givens angles), six tasks. Aligned teacher uses angle vector m_t φ; stress uses independent angle vectors. Fit task maps on 512 examples/task and test on 1024. Development seeds 35/53; initial fresh 137/243/347/457 (payload replay invalid; preserved); corrected audit 163/269/367/463. Measure MSE, orthogonality, inverse-cycle error, actual bytes, active MACs, and CPU transform wall time. Zero optimizer updates.

## Gates

PASS: Mirror within 1.10x independent BOFT error, <=80% bytes, <1e-10 orthogonality/cycle error on >=3/4 aligned seeds, and lower bytes than the simple factorized-angle control. FAIL: no Pareto improvement over that simple control or >1.25x independent BOFT MSE on >=3/4.

Charge W, independent BOFT vectors, shared φ/task m, and all reconstruction state in uncompressed NPZ bytes. CPU timing is exploratory.

## H / T / D / C / U

FACT: pending.  
INTERPRETATION: pending.  
HYPOTHESIS: pending.  
COUNTER-HYPOTHESIS: pending.  
UNCONFIRMED: pending.  
BOUNDARY: synthetic linear butterfly transforms; no LM/diffusion/training/capacity claim.

## Results — H / T / D / C / U

FACT: Corrected fresh seeds 163, 269, 367, 463: aligned orbit independent BOFT MSE averaged 4.98e-25 at 1,582 bytes. Mirror and the ordinary factorized-angle control both had 7.26e-31 MSE at 1,150 bytes. In independent-angle stress, both factorized models had MSE 0.06944 while independent BOFT was numerical zero. The shared single BOFT also retained larger aligned error in development. Orthogonality and cycle checks passed at numerical precision. The first fresh set was retained but excluded after roundtrip verification found the evaluated optimized state differed from serialized initial coordinates; corrected audit used new seeds. Payload replay tests now pass.

INTERPRETATION: A shared BOFT angle vector plus task scalar compresses an aligned task orbit by 27.3% vs independent BOFT, but the direct ordinary factorization is exactly the same function and bytes. Unrelated angle matrices have a clear error boundary.

HYPOTHESIS: Butterfly structure can provide efficient transform atoms, while factorization rank controls which task variations remain recoverable.

COUNTER-HYPOTHESIS: The aligned synthetic teacher directly uses the shared angle factorization and overstates transfer to trained adapters.

UNCONFIRMED: PEFT training, downstream neural quality, higher rank/codebooks, optimized runtime and generalization to diffusion/LM models.

Decision: FAIL for Mirror-specific advantage; preserve the aligned compression result as ordinary low-rank factorization over BOFT angles.


## Reconciled decision

The post-fit fresh audit is FAIL for Mirror-specific value because the ordinary scalar-times-shared-angle factorization exactly matches Mirror in function and bytes. A separately preregistered neutral-initialization trained screen also failed at development and did not open its fresh worlds. These protocols answer different questions and are not pooled; see `RECONCILIATION.md`.
