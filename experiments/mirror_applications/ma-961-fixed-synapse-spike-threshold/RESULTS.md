# MA-961 results — fixed-synapse spiking threshold task views

## H — Hypothesis

An affine task-index coordinate can replace per-task neuron thresholds with less serialized state while retaining threshold-classifier quality. Irregular threshold worlds should reveal the need for private residual values.

## T — Test

**Fact.** CPU deterministic integrate-and-fire simulation, 3 fresh worlds (96121, 96123, 96127), 16 tasks/world, 4,096 Gaussian test examples/task. One shared scalar synapse and one LIF unit, 8 steps, leak 0.9, reset-to-zero, input bias 1.5. Each task is a binary threshold decision `x > c_t`; task boundaries are fresh random draws sorted across task IDs. Compared FP32 threshold bank, direct min/max affine 4-bit codes, least-squares affine task-index threshold code, and affine code plus a full private residual table. Actual `torch.save` inference payload bytes include synapse, threshold/code, calibration state, and metadata. Amendment 01 wrote unmatched metrics then errored while printing; those files are preserved and excluded. Amendment 02 used three new seeds.

## Results

**Fact.** Means over all task×world rows:

| Condition | Accuracy | Mean spikes/example | Serialized payload bytes |
|---|---:|---:|---:|
| Independent FP32 thresholds | 1.0000 | 0.8083 | 2,101 |
| Direct 4-bit threshold quantization | 0.9889 | 0.8050 | 2,299 |
| Affine task-index Mirror code | 0.9721 | 0.7589 | 2,581 |
| Affine code + full private residual | 1.0000 | 0.8083 | 2,809 |

The affine code failed the frozen 99% accuracy gate and was 480 B larger than the FP32 baseline. Direct 4-bit coding narrowly missed 99% mean accuracy and was 198 B larger. Exact private residuals restored outputs but increased payload by 708 B. The number of spikes changed with threshold reconstruction; this is an operation-count proxy, not joules or measured hardware energy. Per-world/task results and threshold tables are in `RESULTS_CORE.csv` and `source/results.json`.

**Interpretation.** At this tiny scale, serialization metadata outweighed the nominal threshold-code savings. Irregular task thresholds were poorly represented by one affine task coordinate; restoring them required a full residual table. Direct quantization was closer in quality, but its actual serialized payload still exceeded the native FP32 bank.

## D — Decision

**FAIL (scoped synthetic threshold screen).** Neither affine Mirror code nor direct 4-bit threshold codes passed the preregistered combined quality/storage gate. No energy or trained-SNN advantage is established.

## C — Strongest counter-hypothesis

The sixteen-task fixture is too small for codebook metadata to amortize. A larger task bank with genuinely smooth threshold structure could move the full-payload frontier; the affine approximation also assumes task-index regularity that random worlds do not provide.

## U — Unconfirmed

No STL-SNN implementation, gradient-trained network, multi-neuron event model, real event dataset, hardware spike energy, wall-clock serving benchmark, or generalization to unseen task thresholds was tested. The task functions are one-dimensional decision boundaries, not independent classifiers.

## Verification

The corrected amendment 02 script was syntax-checked and run. The FP32 model reproduces its teacher threshold labels exactly on every row. Inference payloads were serialized for each condition/world. The first aborted output is retained separately and excluded from reported metrics.
