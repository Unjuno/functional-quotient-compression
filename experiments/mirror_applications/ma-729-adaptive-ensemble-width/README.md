# MA-729 — uncertainty-triggered adaptive Mirror ensemble width

Status: SCREENING

## Hypothesis

H: Disagreement-triggered activation of additional Mirror Views preserves most four-member ensemble quality while reducing average active compute, and improves the risk/coverage/compute frontier over fixed-width and ordinary early-exit ensembles.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m`, a 16-angle member code over one shared 64-dimensional linear predictor, so the same physical vector generates four logical member functions and only uncertain inputs pay for more active Views.

Prior art: Laplace (PA183), Packed-Ensembles (PA184), Rank-1 BNN (PA41), MIMO (PA42). Controls include independent full member vectors (same logical functions), fixed width 1/2/4, shared native Givens view bank, and confidence-threshold early exit. This is a synthetic mechanism screen.

## Frozen protocol

D=64, four member functions, 16 fixed disjoint coordinate-plane rotations/member, 2,048 train-free development and 4,096 held-out examples; seed 72901. One shared vector `w` generates members through per-view angle codes. Labels are majority vote of all four teachers. Compare fixed k=1/2/4 and adaptive stopping after each member when ensemble probability margin exceeds threshold τ in {.10,.20,.30,.40}; if final margin remains below τ, report selective abstention. Report risk, coverage, NLL, diversity, average active members, active MAC proxy, CPU latency distribution and actual serialized model+policy bytes. Native angle-coded bank is a mandatory exact baseline.

### Gates

PASS requires adaptive coverage >=.95, selective risk no more than .02 above fixed width-4, >=25% lower average active compute than fixed width-4, and a better compute/quality frontier than confidence-only early exit. Mirror-specific value also requires beating the native angle-coded bank. Exact function/state alias or failure of the quality/coverage gate is FAIL; no audit after development failure.

## Scope

Synthetic majority-vote linear ensemble only. No trained models, natural uncertainty calibration, selective classification benchmark or production latency claim.

## Results — development seed 72901

**FACT:** Fixed-4 independent member vectors use 1,069 B, with risk .0942 and accuracy .9058. Shared angle-coded fixed-4 uses 1,382 B for identical predictions; the adaptive code/policy payload is 1,570–1,571 B. Thresholds .1/.2/.3/.4 yield disagreement-policy coverage .997/.980/.969/.924, selective risk .233/.228/.222/.219, accuracy .767/.771/.777/.785 and mean active views 1.058/1.122/1.168/1.276. None meets frozen coverage+risk gates. Confidence-only control produces the same operating points (disagreement criterion is redundant here). Active MAC proxy falls from 448 at fixed-4 to 118–142, but per-example p50 latency for adaptive disagreement is ~.039–.040 ms versus .160 ms shared fixed-4 and .012 ms independent fixed-4; transforms are reconstructed during each view evaluation. Mirror state is 29.3% larger than independent vectors at fixed-4 and adaptive policy metadata increases it further.

**INTERPRETATION:** Adaptive width lowers arithmetic and measured latency relative to shared fixed-4 only by accepting substantially higher selective risk. At equal quality, the independent fixed-4 bank is both smaller and faster than the shared angle code. Disagreement did not improve over confidence-only stopping.

**HYPOTHESIS:** An uncertainty signal calibrated to member error, rather than confidence/disagreement alone, may improve selective risk; this run does not support that claim.

**D:** FAIL for the registered quality/storage/computation frontier; audit unopened.

**C:** The synthetic label is a majority vote of the same four linear Views, and only one seed is used. This task may misrepresent real epistemic uncertainty; still, the frozen test directly shows no gain under this registered mechanism.

**U:** Natural benchmark risk/coverage, calibrated neural uncertainties, fresh worlds, actual GPU launch/latency and whether another disagreement estimator can retain quality at low active width.

### H / T / D / C / U

- **H:** Adaptive Mirror width can save compute on easy inputs while preserving risk and coverage versus fixed-width ensembles.
- **T:** D=64, four shared-base Givens member views, 2,048 development and 4,096 held-out synthetic points, seed 72901; fixed widths, confidence-only and disagreement stopping.
- **D:** FAIL; no threshold meets risk/coverage gates; adaptive state costs more bytes than independent models; disagreement equals confidence-only.
- **C:** Simplified synthetic teacher and single world.
- **U:** Natural calibration, fresh seeds and optimized inference kernels.
