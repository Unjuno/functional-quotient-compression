# MA-1024 results

## H — Hypothesis

A domain phase on one shared four-token CLIP CoOp context can preserve base-to-novel accuracy over four image domains while using less prompt state than independent domain contexts, and provide value beyond an ordinary scalar Givens conditioner.

## T — Test

- Frozen `openai/clip-vit-base-patch32` (151,277,313 parameters) and CIFAR-10 (`uoft-cs/cifar10`).
- Base train classes: airplane, automobile, bird, cat, deer, dog; 4 labeled train images/class. Novel audit classes: frog, horse, ship, truck.
- Four deterministic domains: original, grayscale, brightness 0.65 and Gaussian blur radius 1.0.
- Frozen prompt length: 4 context tokens. Thirty Adam updates at lr 0.002 for shared CoOp, independent per-domain CoOp and Mirror. Official CIFAR-10 test images (8 per class) were accessed only after all prompt states were frozen.
- Compared zero-shot CLIP, shared CoOp, independent domain CoOp, Mirror Givens phase and the matched native scalar Givens control. The latter shares identical state and outputs by construction.
- CPU only. Image feature extraction: 18.15 s total (4.29 s train, 13.86 s audit); prompt training: 162.04 s total; full run: 189.74 s. Text-prompt evaluation took 1.16–1.20 s/model. The inference operation includes the 151M-parameter CLIP backbone; prompts are small relative to it.
- Complete bundle bytes include CLIP weights, configuration, tokenizer vocabulary/merges, tokenizer JSON and image preprocessing config plus the serialized prompt payload. Prompt serializer is JSON metadata plus contiguous little-endian float32 arrays.

## Results

| Method | Base accuracy | Novel accuracy | Harmonic mean | Prompt bytes | Full inference bundle |
|---|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 76.56% | 83.59% | 79.92% | 358 B | 608,863,938 B |
| Shared CoOp | 74.48% | **88.28%** | **80.79%** | 8,612 B | 608,872,192 B |
| Independent domain CoOp | 73.96% | 85.94% | 79.50% | 33,191 B | 608,896,771 B |
| Mirror Givens phase | **78.13%** | 77.34% | 77.73% | 8,685 B | 608,872,265 B |
| Native scalar Givens control | **78.13%** | 77.34% | 77.73% | 8,685 B | 608,872,265 B |

Mirror uses 73.8% fewer prompt bytes than independent domain CoOp, but its novel accuracy is 8.59 percentage points lower and its base/novel harmonic mean is 1.77 points lower, missing the preregistered 1-point quality tolerance. Shared CoOp has 10.94 points higher novel accuracy and a 3.06-point higher harmonic mean while using 73 fewer prompt bytes. Including the shared CLIP files, Mirror saves only 24,506 B from a 608.9 MB bundle (0.0040%) versus independent contexts.

## D — Decision: FAIL

**FACT:** The Mirror and native scalar Givens controls have byte-identical prompt payloads and identical accuracy. Independent contexts are larger, but Mirror misses their quality tolerance. Shared CoOp is smaller and has better novel-class performance. Fresh test examples were scored once after prompt freeze; deterministic replay with the same indices reproduced all metrics and payload sizes exactly. Three tests pass. The CLIP checkpoint and selected CIFAR images are identified by hashes in `runs/development.json`; the official test split has been opened and was not used to tune.

**INTERPRETATION:** The phase family compresses an independent domain prompt bank on this small dataset, but loses transfer to novel classes. A common CoOp prompt generalizes better, and ordinary scalar Givens explains the entire Mirror effect. Counting the frozen backbone makes the end-to-end storage reduction negligible.

**HYPOTHESIS:** This screen fails to establish a useful Mirror advantage for CoOp domain prompts. It suggests a quality/parameter boundary: independent domain contexts can specialize, while a four-token shared basis with one angle per domain does not preserve novel-class transfer.

**C — Strongest counter-hypothesis:** Four-shot training and four synthetic image transforms may underfit domain-conditioned context; larger natural domain datasets or a private residual could recover novel performance.

**U — Unconfirmed:** Other CLIP backbones, larger class vocabularies, multiple datasets, natural domain shifts, CoCoOp/MaPLe comparisons, private-residual frontier and external fresh random seeds.
