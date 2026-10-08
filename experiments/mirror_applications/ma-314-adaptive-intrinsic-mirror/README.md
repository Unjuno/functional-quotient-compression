# MA-314 — adaptive intrinsic-dimension Mirror allocation

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / QUALITY / COMPUTE / RUNTIME  
Base commit: `9ca5c968b5ec9a285b11140f08a0f573c8e07cf3`  
Prior art: PA33 intrinsic-dimensional fine-tuning; MA-311 and MA-312 shared intrinsic Mirror screens

## H — falsifiable hypothesis

Across task functions whose required intrinsic dimensions vary, a validation-selected prefix of shared two-dimensional Mirror planes will allocate fewer actual serialized bytes than adaptive random-subspace (SAID) and direct coefficient vectors, while matching held-out quality and choosing the smallest sufficient dimension.

> **Mirror insertion:** this experiment adds per-task Mirror angles `m_t` and an active-dimension address to a shared intrinsic update basis so that tasks requiring different numbers of coordinate pairs can use different logical functions without storing a full coefficient vector or private weight vector per task.

## T — frozen mechanism screen

One shared 16D linear predictor and a paid random orthonormal 16×8 task-update basis serve 48 task IDs. Target updates use 1, 2, or 4 active coordinate pairs (intrinsic dimensions 2, 4, 8), balanced across tasks; each active pair has a shared fixed radius and task-specific angle. Tasks receive 64 support, 64 validation, and 128 test examples. Fit codes on support only; choose the smallest active dimension whose validation normalized MSE is at most `1e-4`; report test metrics only after that choice. No optimizer updates; task IDs are supplied.

Controls: hard tie; adaptive PA33-style least-squares random-subspace coordinates (FP32); adaptive direct FP16 coefficient codes on the same basis; fixed-dimension SAID-8; and independent full 16D task weights. Mirror fits pair angles with four coordinate-descent sweeps on a frozen 720-point grid. Every method pays the shared predictor and full maximum basis, active-dimension metadata, codes, radii and deterministic ZIP/NPY headers. Development seeds 31401–31403; fresh seeds 31411–31413 are sealed unless all development gates pass. No hyperparameter is tunable.

## Gates

**Promising / open fresh:** in each development world, Mirror test normalized MSE ≤`1e-4`, chooses the registered minimum dimension for at least 95% of tasks, and actual bytes are ≤90% of both adaptive FP16 coefficients and adaptive SAID; per-dimension test error is within `1e-5` absolute normalized MSE of adaptive FP16 coefficients.  
**Fail:** any development byte/quality/dimension gate fails. A simple coefficient match also fails Mirror-specific value.  
**Not established:** replay, serialization or split-integrity verification fails.

## C / U

**Strongest counter-hypothesis:** an angle is a polar encoding of two ordinary coefficients on a deliberately aligned fixed-radius orbit; a compact FP16 coefficient vector may be smaller or more accurate, and validation thresholding may select the same dimensions without Mirror. The varying-dimension task bank can therefore show allocation utility while falsifying Mirror-specific value.

**Unconfirmed:** learned bases, noisy/natural tasks, gradient-trained models, learned routing, capacity at convergence, and tasks requiring directions outside the shared basis (private-state frontier).

## Results — fresh adaptive-dimension screen

### H / T / D / C / U

**H:** validation-selected active Mirror dimension should reduce actual payload over adaptive random-subspace and direct coefficient controls while retaining test quality and choosing the minimum sufficient dimension.

**T:** 48 tasks with balanced true dimensions 2/4/8; shared 16D predictor and full paid random 16×8 orthonormal basis; each task received 64 support, 64 validation and 128 test examples. Codes were fit on support, smallest dimension under validation normalized MSE `1e-4` was selected, and test was scored once. Development 31401–31403 passed all frozen gates before fresh 31411–31413 were run. Controls were tied, adaptive FP32 SAID, adaptive FP16 direct coefficients, fixed SAID-8 and independent full vectors. Zero gradient updates, oracle task IDs.

**D: PROMISING, narrowly for this aligned synthetic storage/quality point.**

**FACT:** On all three fresh seeds, Mirror selected the exact true dimension for 48/48 tasks and had maximum task normalized MSE at most `6.75e-6` (world mean `1.20e-6`–`2.09e-6`). Actual payload was **1,996B**, versus **2,220B** adaptive FP16 coefficients (10.1% fewer bytes), **2,668B** adaptive SAID, **3,308B** fixed SAID-8, and **3,320B** independent vectors. Mirror remained within the registered `1e-5` absolute normalized-error margin of adaptive coefficients, though coefficients were about 100–300× more accurate in normalized MSE. Hard tying used 310B but had mean normalized MSE 0.146–0.325. All shared predictor, full maximum basis, radii, task dimension tags, codes and archive headers were charged.

Mirror's fit proxy was **371,589,120** operations versus **286,080** for adaptive coefficient least squares (~1,299×); its active inference proxy was **104.7** versus **90.7** operations/example (~15.4% higher). Median fresh CPU throughput was 3.95M examples/s Mirror and 6.46M direct coefficient, with timing sensitive to this small synthetic workload; there is no compute/runtime win. Adaptive dimension selection itself recovered the registered 2/4/8 frontier in every fresh world. Four tests passed and all 36 development/fresh result rows replayed exact metric, byte and payload hash.

**INTERPRETATION:** a variable active dimension can be selected from validation and gives a useful storage point for this fixed-radius, shared-basis task family. The representation is a polar encoding of ordinary coefficient pairs, so this is a narrow Mirror-coded payload reduction, not evidence that Mirror adds function capacity or beats coefficient adaptation on compute or quality precision.

**C — strongest counter-hypothesis:** target updates were deliberately generated from the same fixed-radius polar planes. Different radii, off-basis directions, learned bases, task-discovery cost, or packed coefficients could erase the byte margin. The simpler FP16 coefficients are more accurate and vastly cheaper to fit.

**U:** private-state frontier outside the basis, noisy/natural task distributions, learned projections, nonlinear/deep models, learned routing, and near-convergence capacity remain untested.

FACT: see `RESULTS_CORE.csv`, `DEV_RESULTS.csv`, `FRESH_RESULTS.csv`, `source/dev_gate.json`, `source/fresh_gate.json`, and `source/replay_verification.json`.  
INTERPRETATION: dimension allocation passes; Mirror improves only the aligned payload point.  
HYPOTHESIS: task-specific variable dimension can reduce storage when required functions occupy nested shared subspaces.  
BOUNDARY: synthetic post-fit task bank; no capacity, language, private-state, or broad runtime claim.

## Decision

FACT: see `RESULTS_CORE.csv`, `FRESH_RESULTS.csv` (if opened) and `VERIFICATION.json`.  
INTERPRETATION: separate adaptive dimension selection from the incremental value of polar Mirror codes; charge all serialized state and fit/inference work.  
HYPOTHESIS: active intrinsic dimension may be allocated per task with a compact coordinate.  
BOUNDARY: synthetic, post-fit, task-ID-supervised linear task bank; no capacity or language claim.
