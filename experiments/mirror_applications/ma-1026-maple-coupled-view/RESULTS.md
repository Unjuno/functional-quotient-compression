# MA-1026 — Coupled visual/text MaPLe prompt views

Status: **FAIL** for the preregistered Mirror-specific gate. This is a bounded CIFAR-10 CPU screen using an explicitly reduced MaPLe architecture; it is not an official MaPLe reproduction.

## H — Hypothesis

Separate visual and text Givens coordinates over one shared coupled MaPLe prompt state preserve domain×class and base-to-novel quality while reducing prompt bytes against independent per-domain MaPLe; they also beat a matched rank-1 additive control in H-mean by at least 1 percentage point.

## T — Trial

- Read and inspected PA314's official implementation at commit `69bce21ae8eda80ad6187534b2dce09cf6c59e17`. Implemented the official coupled prompt structure: text shallow context projected to visual prompt tokens, plus one deeper text prompt and its learned visual projection.
- Frozen `openai/clip-vit-base-patch32`, 151,277,313 parameters, with prompt tokens at both shallow input and after transformer layer 0. The registered prior's ViT-B/16, context depth 9, five-epoch configuration was reduced to ViT-B/32 and depth 2 (one deep prompt) for this CPU screen.
- CIFAR-10 base classes 0–5 and novel classes 6–9. Four image domains: original, grayscale, brightness 0.65 and Gaussian blur radius 1.0.
- Four train images/base class and four official test images/class. Train and test IDs are disjoint from MA-1024 and MA-1025. The test pixels were opened after all prompt states were frozen; 40 image hashes are recorded. Test audit contains 160 class×domain examples.
- For compositional evaluation, training omitted all images from six predeclared base class×domain cells (`domain = class_id mod 4`). Each class and each domain appeared in other training combinations. This leaves 72 unique train image×domain examples.
- Five comparisons: zero-shot CLIP; one shared coupled MaPLe state; independent coupled MaPLe prompts/projections per domain; shared prompts plus two per-domain Givens angles (one text, one visual); and matched per-domain rank-1 additive bases for the text and visual streams.
- Each learned method received 20 Adam updates at lr .001 and batch size 4 in a fixed round-robin domain schedule, consuming 20 image examples per domain per method. The independent method therefore updates each independent domain state only five times; all update and example counts are reported.
- Inference payloads are self-describing JSON manifests plus little-endian float32 arrays. Full bundles include all cached CLIP weights, configuration, tokenizer and processor files. The reported prompt-view operation counts exclude the much larger frozen CLIP encoder.

## D — Decision

**FAIL.** Mirror satisfies the registered comparison to independent prompts: novel accuracy is equal and H-mean is higher in this small audit, while prompt state is 75.0% smaller. However, it misses the required H-mean advantage over the rank-1 additive control by 1 percentage point: Mirror is 1.28 pp worse. Its H-mean is also 1.28 pp below ordinary shared MaPLe. The view code therefore adds no demonstrated functional benefit over hard sharing or simple additive conditioning.

### Facts

| Method | Base acc. | Novel acc. | H-mean | Held-out base cells | Prompt bytes | Full bundle bytes | Eval wall / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 80.21% | 84.38% | 82.24% | 83.33% | 475 B | 608,864,055 B | 0.05048 s |
| Shared MaPLe | 78.12% | 87.50% | 82.55% | 79.17% | 3,161,012 B | 612,024,592 B | 0.05168 s |
| Independent per-domain MaPLe | 61.46% | 85.94% | 71.67% | 50.00% | 12,641,981 B | 621,505,561 B | 0.05225 s |
| Mirror paired Givens | 77.08% | 85.94% | 81.27% | 79.17% | 3,161,136 B | 612,024,716 B | 0.05234 s |
| Additive rank-1 multimodal | 78.12% | 87.50% | 82.55% | 79.17% | 3,181,754 B | 612,045,334 B | 0.05259 s |

- The novel audit has 64 examples total (16 per class). Mirror and independent MaPLe both classify 55/64 correctly. The held-out base domain×class score uses 24 examples; Mirror, shared MaPLe and additive each score 19/24.
- Mirror prompt state is 9,480,845 B smaller than the independent prompt bank (75.0%). The complete bundle is 9,480,845 B smaller because the same 608,863,580 B frozen backbone dominates. Full-model storage falls about 1.53% versus the independent bundle.
- Mirror is 124 B larger than the shared MaPLe payload, with lower measured base, novel and H-mean accuracy. The additive control is 20,618 B larger than Mirror, but has exactly the same aggregate accuracies as hard sharing and a 1.28 pp higher H-mean than Mirror.
- Prompt-view proxy per image: 15,360 scalar FLOPs for two-stream Givens rotations versus 5,120 MACs for rank-1 additive prompt bases. Full audit wall is 8.37 s for Mirror (160 examples), 8.41 s additive, 8.27 s shared and 8.36 s independent. These single-run CPU timings are close and do not support a runtime claim.
- Training wall was 15.82 s shared, 15.31 s independent, 15.49 s Mirror and 15.14 s additive. Total run wall was 107.75 s.
- Frozen asset hashes, all prompt payload hashes, full-bundle arithmetic, audit hashes, previous-experiment index disjointness and exact serialized-prompt metric replay passed. Five unit tests passed.

### Interpretations

- The paired prompt mechanism can be stored with far less state than four independent prompt banks, but hard sharing already gives equal or better measured quality at effectively the same size.
- The independent reference is undertrained per domain under the fixed total-update budget (five updates/domain), so its lower base score should not be treated as a quality upper bound. It only supports a fixed-budget state comparison.
- The Mirror transformations cost more prompt-operation FLOPs than the low-rank control and provide no observed held-out-cell benefit beyond ordinary sharing.
- Because the CLIP backbone is over 608 MB, a 9.48 MB complete-bundle change is modest; all methods retain the same backbone.

### Hypotheses for follow-up

- Longer per-domain training might improve independent prompts and change the relative quality frontier; that would be a separately registered/amended budget.
- More tokens, deeper prompts, larger audits and additional seeds could change the result, but this screen provides no evidence that Givens geometry would then beat low-rank additive conditioning.

## C — Strongest counter-hypothesis

The independent MaPLe bank received only five updates per domain and the audit uses just four images per class per domain, while every Mirror code is trained for only 20 total updates. The measured storage/quality relation is consequently a fixed-budget, low-data signal. Still, the Mirror did not outperform its matched additive control or the shared MaPLe state.

## U — Unconfirmed

- Official ViT-B/16, depth-9 and five-epoch MaPLe behavior; larger shot counts, seeds, datasets and natural domain shifts.
- Whether longer domain-specific training would preserve Mirror's prompt-state saving at matched quality.
- Any general MaPLe or vision-language model claim, GPU throughput result, or capacity increase.

## Fact / Interpretation / Hypothesis separation

The table, model/data hashes, update counts, payload bytes and replay results are **Facts**. The stated consequences of matched scores, shared-state dominance and short independent training are **Interpretations**. Longer training/deeper prompts are **Hypotheses**.
