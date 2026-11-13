# MA-961 corrected results — fixed-synapse spiking threshold task views

## H — Hypothesis

An affine task-index coordinate can replace independent per-task thresholds with lower serialized payload while retaining threshold-classifier accuracy. Direct 4-bit threshold quantization is the strongest simple storage control; irregular thresholds may require private residuals.

## T — Test

**Fact.** CPU deterministic integrate-and-fire simulation, three new fresh worlds (96131, 96133, 96137), 16 tasks/world, 4,096 Gaussian test examples/task. One shared scalar synapse and one LIF unit, 8 steps, leak 0.9, zero reset, input bias 1.5. Each task is `x > c_t`; task boundaries are fresh sorted random draws. Compared FP32 thresholds, direct min/max 4-bit codes, least-squares affine task-index code, and affine code plus a full private residual table. Amendment 04 packed two 4-bit codes per byte. Every inference threshold was decoded from the serialized state loaded with `weights_only=True` and that decoded value was used for scoring. All state tensors matched their pre-serialization values exactly. Payload bytes include tensor archive, metadata, min/max range or affine parameters, task indices, and shared synapse.

Amendment 02 metrics that scored in-memory float64 reconstructions are superseded. Amendment 03 also reused those prior seeds by implementation error; its output is preserved and excluded. Amendment 04 ran the corrected scoring code on new seeds.

## Results

**Fact.** Means over 48 task×world rows:

| Condition | Accuracy | Mean spikes/example | Serialized payload bytes |
|---|---:|---:|---:|
| Independent FP32 thresholds | 1.0000 | 1.1145 | 2,101 |
| Direct packed 4-bit threshold codes | 0.9901 | 1.1018 | 2,363 |
| Affine task-index Mirror code | 0.9588 | 1.0851 | 2,581 |
| Affine code + private residual | 1.0000 | 1.1145 | 2,809 |

The affine code failed the frozen quality gate and used 480 B more than FP32 (+22.8%). Direct 4-bit coding averaged 99.01% accuracy but still used 262 B more (+12.5%), missing the 20% actual-payload saving gate. Full residuals recovered the teacher exactly but added 708 B (+33.7%). The packed code used 8 bytes for sixteen 4-bit codes; its larger total file includes quantization range and serialization/manifest overhead. Spike counts are simulator operation counts, not joules.

**Interpretation.** In this tiny task bank, neither threshold coding scheme improved the full serialized storage frontier. The affine task coordinate lost quality and cost more bytes; direct quantization retained near-perfect quality but its range and format metadata erased the nominal code savings. An irregular threshold table required the full private residual.

## D — Decision

**FAIL (scoped synthetic threshold screen).** The preregistered affine quality/storage gate failed. Direct packed int4 was close in quality but larger in actual payload than FP32. No STL-SNN, energy, or learned-network advantage is established.

## C — Strongest counter-hypothesis

The sixteen-task model is too small for codebook/range metadata to amortize. A much larger bank might reduce per-task storage, though an arithmetic progression would still be an ordinary native scalar parameterization rather than Mirror-specific value.

## U — Unconfirmed

No STL-SNN implementation, gradient-trained network, multi-neuron event model, real event dataset, hardware spike energy, serving benchmark, or unseen-task generalization was tested. These are one-dimensional thresholds, not independent classifiers.

## Verification

The corrected run used seeds 96131, 96133, 96137. It serializes each condition, reloads all tensor states, decodes its actual threshold representation, and scores using those decoded thresholds. Direct int4 nibble packing was round-tripped. Superseded runs are retained as `source/amendment02_unloaded_scoring.*` and `source/amendment03_reused_seeds.*`.
