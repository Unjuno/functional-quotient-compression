# MA-966 results — fine event timing under shared Mirror phase views

Status: **FAIL for the preregistered Mirror-specific screen**. Broader use of learned temporal views in pretrained event models remains unestablished.

## H — hypothesis

A shared event encoder with one compact phase coordinate per temporal window would improve sign decoding on unseen Vernier offsets over the same shared encoder, retain exact timestamp access, and use substantially fewer serialized bytes than four independent encoders.

## T — execution

**Fact.** We used the public D1 Vernier acquisition from the Event-Based Tiny Images dataset (Zenodo record 20973757), extracting only `events.pkl` and `labels.pkl` with HTTP byte ranges from the 715,632,615-byte archive. The extracted files are SHA-256 recorded in `source/DATA_PROVENANCE.json`. The official loader rescales timestamps from microseconds to milliseconds; our extraction follows that scale. There were 708 complete trials: 202 train trials (labels 13, 14, 26, 27; offsets ±6/±7), 101 development trials (labels 15, 25; ±5), and 405 sealed fresh trials (labels 16–19 and 21–24; offsets ±1..±4). Four chronological 48-event windows began at 80, 100, 120, and 140 ms. Each original trial remained in one split.

**Fact.** The from-scratch CPU adapter was a one-layer, 32-wide Transformer with four heads. Conditions were shared raw-time, shared fixed periodic sin/cos time features, shared learned per-window Mirror phase, native per-window phase/time offset, phase-only without raw time, and four independent encoders. AdamW ran for up to 30 epochs with development early stopping. Fresh optimizer seeds were 96611, 96613, and 96617. Three seeds are repeated fits on one acquisition, not independent data worlds. We also computed the official-style raw-event Ridge diagnostic. No MNIST-pretrained PointTransformer checkpoint was used.

**Fact.** Each model payload is the actual `torch.save` byte count of its inference state dict plus condition/config metadata. Fresh predictions were computed after loading that serialized state. Exact phase-to-native-offset function replay had maximum absolute logit difference **0.0**. State reload reproduced development accuracy within `3e-8`.

## Fresh quality and storage

| Condition | Macro sign accuracy, offsets 1–4 (mean ± seed SD) | Offset 1 | Offset 2 | Offset 3 | Offset 4 | Payload bytes |
|---|---:|---:|---:|---:|---:|---:|
| Shared raw time | 81.93% ± 0.89 | 66.18% | 76.49% | 90.68% | 94.39% | 41,773 |
| Shared fixed periodic features | 83.14% ± 1.21 | 68.55% | 77.39% | 91.17% | 95.46% | 42,029 |
| Shared Mirror phase | 82.08% ± 1.24 | 66.58% | 76.82% | 90.10% | 94.80% | 42,346 |
| Native phase/time offset | 82.08% ± 1.24 | 66.58% | 76.82% | 90.10% | 94.80% | 42,410 |
| Phase-only, raw time removed | 80.82% ± 0.84 | 66.75% | 74.26% | 88.78% | 93.48% | 41,901 |
| Four independent encoders | 83.56% ± 0.53 | 67.65% | 79.95% | 91.17% | 95.46% | 166,923 |

The raw-event Ridge diagnostic averaged 49.5% across fresh offsets, near chance. It is only a diagnostic on flattened event inputs, not the paper's pretrained-feature benchmark.

**Fact.** The Mirror phase model uses 25.4% of the independent payload bytes (74.6% fewer), but it is 573 B larger than the shared raw-time model and 317 B larger than the fixed periodic model. Its three-seed mean quality is 0.14 percentage points above raw shared time, 1.07 points below fixed periodic features, and 1.48 points below independent encoders. Mirror and native offset had identical per-seed accuracy at every offset.

**Interpretation.** The large byte reduction versus independent encoders comes from sharing the Transformer, not from the phase coordinate. The phase coordinate adds bytes without improving the strong periodic or native-offset control. The 64-byte size difference between Mirror and native-offset payloads is serialization metadata/key length, not model capacity. The exact alias check confirms that rotating the sin/cos pair by `m` is the same function as a native time offset before periodic encoding.

## Runtime and timing audit

**Fact.** Training wall time averaged 10.9 s for Mirror, 14.4 s for native offset, and 14.6 s for independent encoders in this CPU run; these are noisy single-process times. A balanced development-only runtime repeat measured medians of 0.0588 s per 404-window batch for Mirror and 0.0662 s for native offset; the condition order and host load materially affected timing, so no runtime advantage is claimed. All methods re-run the encoder for each window; shared weights do not reuse computed states.

**Fact.** On the raw-time seed 96611 jitter check, macro accuracy was 82.88%, 82.14%, 83.00%, and 82.07% for timestamp jitter σ=0.25, 0.5, 1, and 2 ms. This short check is not monotonic and is not evidence of temporal robustness or an energy result. The phase-only ablation was 1.12 points below raw shared time on the fresh mean, with no significance test.

## D — decision

**FAIL.** The registered Mirror-specific gate fails: Mirror does not improve by 2 points over raw shared time, does not beat the equal-function native offset control, and loses to fixed periodic features. It does fit within 80% of independent-encoder bytes and comes within 2 points of independent quality, but those benefits are also present in ordinary shared encoders and do not isolate Mirror value.

## C — strongest counter-hypothesis

**Interpretation.** The paper's hyperacuity result depends on an MNIST-pretrained PointTransformer representation. This small from-scratch model on one D1 acquisition may not reproduce that representation; the raw-event Ridge diagnostic is also weak. Therefore the experiment cannot rule out phase-conditioned views on pretrained event models. The exact algebraic alias to native phase/time offsets remains established for the tested insertion.

## U — unresolved

- Replication across independent acquisition sessions, distances, or event sensors.
- The published MNIST-pretrained PointTransformer Vernier feature result.
- End-to-end learned m, downstream detection, GPU latency, energy, and hardware event throughput.
- Whether a richer non-affine phase coordinate can beat native FiLM, periodic encoding, or time-offset controls.

## Evidence separation

- **Fact:** the measured dataset counts, accuracies, bytes, runtime samples, hashes, and zero-logit alias error above.
- **Interpretation:** shared periodic conditioning can help this small task; the Mirror coordinate supplies no measured gain beyond its ordinary offset alias.
- **Hypothesis:** nonlinear task-specific phase codes could matter in a stronger pretrained encoder; this screen neither demonstrates nor rejects that broader possibility.
