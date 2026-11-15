# MA-1033 — Class × domain prompt composition

Status: **FAIL**. This bounded CPU screen does not support a broad VLM claim.

## H — Hypothesis

A product-form class-by-domain Givens coordinate over one shared coupled CLIP prompt basis will synthesize useful prompt views for withheld Cartesian class-domain cells, while reducing storage against independent cell prompts and beating a same-size additive control.

## T — Trial

- Frozen `openai/clip-vit-base-patch32` (151,277,313 parameters; 608,863,580 B cached backbone/config/tokenizer/processor payload) with a two-token shallow prompt basis. Prompts traverse all 12 text and vision transformer layers.
- CIFAR-10 classes were evaluated in four deterministic image domains: original, grayscale, brightness 0.65, and Gaussian blur radius 1.0. For every class, training omitted all examples from domain `class_id mod 4`; every class and every domain remained represented in other pairs.
- Eight training images/class, 20 Adam updates/method, batch size 4, fixed round-robin domain schedule. Fresh official-test samples used eight images/class in each of three locked seeds (103301–103303), selected only after all prompt states were trained and frozen. Train/test image indices were disjoint from MA-1024, MA-1025 and MA-1026.
- Controls: zero-shot CLIP, one shared coupled prompt, product-coordinate Mirror, parameter-matched additive class+domain coordinate, and a full independent class×domain context bank. The independent bank's withheld target cells have no positive examples, so it is only a non-compositional storage/seen-cell reference, not an upper bound on withheld-cell quality.
- The learning state is the same across audit seeds; only the fresh audit images vary. Each method uses the same 80 training examples and 20 updates. This measures fresh-image variance, not independent training-seed variance.
- A first development smoke run incorrectly forwarded one layer only and collapsed learned methods to chance. It was excluded; Amendment A1 records it. The corrected all-layer run was rerun before any audit seed was accessed. The corrected protocol/source/data hashes are frozen in `FREEZE.json`.

## D — Decision

**FAIL.** The predeclared storage gate requires at least 25% fewer complete inference-payload bytes than the independent class×domain bank. Mirror saves only 9.39% of learned-prompt payload (163,728 B) and just 0.0268% of the full CLIP bundle (163,728 B of 610,608,122 B). It also does not beat the matched additive control by the required 1 percentage point on held-out cells: mean accuracy is 86.25% for Mirror and 86.67% for additive. The Mirror is 182 B larger than the shared prompt, whose mean held-out accuracy is equal.

### Fresh audit metrics (mean ± sample SD over three test-image seeds)

| Method | Observed pairs | Held-out pairs | Seen/held-out H | All accuracy | Learned payload | Full bundle | Encoder images/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 82.08 ± 6.55% | 83.75 ± 6.96% | 82.90 ± 6.70% | 82.50 ± 6.61% | 306 B | 608,863,886 B | 23.02 ± 3.30 |
| Shared coupled prompt | 86.25 ± 4.41% | 86.25 ± 6.61% | 86.24 ± 5.52% | 86.25 ± 4.96% | 1,580,632 B | 610,444,212 B | 21.48 ± 3.91 |
| Mirror product code | 85.69 ± 4.46% | 86.25 ± 5.45% | 85.97 ± 4.92% | 85.83 ± 4.68% | 1,580,814 B | 610,444,394 B | 22.55 ± 1.22 |
| Additive class + domain | 86.25 ± 4.41% | 86.67 ± 6.88% | 86.45 ± 5.65% | 86.35 ± 5.02% | 1,580,806 B | 610,444,386 B | 24.06 ± 0.26 |
| Independent cell bank | 77.78 ± 4.34% | 50.42 ± 4.73% | 61.16 ± 4.77% | 70.94 ± 4.38% | 1,744,542 B | 610,608,122 B | 23.20 ± 1.61 |

The sample SD reflects only audit image selection. Seed-level held-out accuracy for Mirror versus additive is 80.00% vs 78.75%, 90.00% vs 91.25%, and 88.75% vs 90.00%; Mirror wins one seed and loses two. The additive method has eight fewer payload bytes than Mirror and higher mean observed and held-out scores. Mirror's 56-byte class/domain coordinate itself does not produce a complete-payload reduction because the shared prompt/projection dominates both payloads.

Each learned method used 80 images and 20 optimizer updates; per-method train wall was 19.4–23.9 s. The exact prompt-view proxy is 614,400 Givens FLOPs over training for Mirror (20 updates × 10 classes × 512 dimensions × 6 scalar operations); additive code performs 200 class-code additions. Full encoder throughput is noisy on this CPU and provides no reliable efficiency advantage. The backbone's transformer compute dominates the coordinate operation cost.

## C — Strongest counter-hypothesis

This is a small, shallow, synthetic-domain, fixed-training-seed screen. Fresh test-image sampling does not estimate training-seed variance; more updates, natural domains, deeper prompt injection, or official CoCoOp/MaPLe training could change the result. The independent-cell reference is under-informed on withheld positives. These limits prevent a broad impossibility claim, but do not rescue the preregistered Mirror gate in this run: shared/additive controls are byte-near and match or exceed Mirror on the same audit samples.

## U — Unconfirmed

- Official CoOp, CoCoOp and depth-9 MaPLe reproduction; natural distribution shifts and larger datasets.
- Training-seed variance, more training images/updates, and larger audit samples.
- Whether other factor geometries or private residual frontiers yield meaningful full-bundle savings.

## Fact / Interpretation / Hypothesis

- **Fact:** Three audit seeds average 86.25% held-out accuracy for Mirror and 86.67% for additive; Mirror learned payload is 1,580,814 B versus 1,744,542 B independent cell bank; full bundle is 610,444,394 B.
- **Interpretation:** The additional product coordinate yields no demonstrated Pareto improvement over additive or ordinary shared prompting at this byte scale. It adds a marginal functional code but does not create useful multiplicity under the frozen gate.
- **Hypothesis:** Larger prompt banks may have a different private/shared frontier, but any follow-up needs a more efficient basis/control and a training-seed audit.

## H / T / D / C / U

**H:** The registered product class×domain coordinate will synthesize held-out cell functions with fewer bytes than independent cell prompts and outperform same-size additive composition.

**T:** Frozen shallow coupled CLIP prompt screen, CIFAR-10 × four deterministic transforms, one omitted Cartesian cell/class, 20 updates per arm, and three fresh official-test image seeds. Compared zero-shot, shared prompt, Mirror product, additive code, and independent cell state. Exact payloads, source/data hashes, training counts, wall time, and metric replay are recorded.

**D:** FAIL under the frozen storage/quality gate.

**C:** The task is a CPU-scale synthetic-domain screen with one training seed and the full CLIP backbone dominating storage; the independent cell bank is not a held-out quality upper bound.

**U:** Natural-domain transfer, official prompt-learning reproductions, training-seed variance, deeper prompts, and a useful private-residual frontier.
