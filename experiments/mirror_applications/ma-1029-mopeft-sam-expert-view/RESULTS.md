# MA-1029 — SAM feature adapter-bank Mirror views

**Decision: FAIL, scoped to this synthetic SAM ViT-B feature-adapter screen.** This is a paper-inspired MoPEFT-adjacent study, not an official MoPEFT reproduction, MESS result, or natural-image claim.

## H — Hypothesis

A rank-8 adapter shared across three visual domains, with two learned per-domain Givens angles, would remain within 1 mIoU point of independent adapters, reduce serialized adaptation bytes by at least 25%, and beat a matched FiLM control by at least 1 point in at least two fresh worlds.

## T — Run

- Official Segment Anything ViT-B, checkpoint SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`; official source commit `dca509fe793f601edb92606367a655c15ac00fdf`.
- Frozen image encoder, prompt encoder and mask decoder; the trainable modules were attached to the final 256×64×64 image embedding. Images were 1024×1024 for SAM input. Bounding boxes were deterministic, tight visible-object boxes with an 8-pixel margin.
- Three generated domains: textured contrast, low-contrast camouflage, and partial occlusion. Development seed 102900 selected learning rates. The settings were committed before generating any audit examples. Three fresh audit worlds used seeds 102901–102903; each had 2 training and 4 audit images per domain, 60 updates/method/world.
- Controls: zero-shot, independent rank-8 adapters, hard-shared adapter, shared adapter plus two Givens coordinates/domain, shared adapter plus latent FiLM, and a paper-inspired feature-space Adapter/linear-branch/prefix/gate mixture.
- Every adaptation payload was serialized with metadata. Full deployment bytes charge the 375,042,383-byte SAM checkpoint, SAM source, experiment runtime source/config (119,578 bytes), and adaptation payload. The shared frozen image encoder was evaluated once per image; its CPU extraction time is reported separately. Decoder latency excludes this shared encoder cost.

## D — FAIL

### Fact

| Method | Mean audit mIoU | Mean boundary F | Adaptation bytes | Full deployment bytes | Decoder ms/image | Train seconds/world |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0.993670 | 0.997217 | 1,837 | 375,163,798 | 73.98 | 0 |
| Independent adapters | 0.996086 | 0.999883 | 56,959 | 375,218,920 | 71.93 | 11.61 |
| Hard-shared adapter | 0.992902 | 0.993614 | 20,317 | 375,182,278 | 74.04 | 11.13 |
| Mirror Givens | 0.992844 | 0.993719 | 20,505 | 375,182,466 | 74.25 | 11.07 |
| FiLM control | 0.995398 | 0.996846 | 20,885 | 375,182,846 | 71.65 | 11.22 |
| MoPEFT-like mixture | 0.996155 | 0.999706 | 42,117 | 375,204,078 | 75.24 | 12.78 |

Mirror used 64.0% fewer adaptation bytes than independent adapters and stayed within 1 mIoU point in each world. Its 6-coordinate code was 48 bytes inside the serialized adaptation payload accounting. But it did not meet the Mirror-specific quality gate: it exceeded FiLM by 1 point in **zero** worlds. FiLM scored above Mirror in all three worlds while costing only 380 more bytes. Hard sharing used 188 fewer bytes than Mirror and was essentially tied: the largest hard-shared/Mirror difference was 0.0153 mIoU points (seed 102901); seed 102902 was exactly equal to the saved metric precision.

The full-deployment difference between Mirror and independent was only 36,454 bytes (0.0097% of the full bundle), because the shared SAM checkpoint dominates storage. Mirror decoder time (74.25 ms/image) did not improve on hard sharing or FiLM. Shared image-encoder extraction averaged 8.48 s/image during development and was common to all methods.

Per-seed metrics, hashes and exact serialized bytes are in `runs/audit_rows.jsonl`; domain-level rows are in `RESULTS_CORE.csv`.

### Interpretation

Sharing the adapter recovers most of independent-adapter quality at substantially lower *adaptation-state* cost in these generated tasks. The additional Givens coordinates do not earn their bytes: a hard-shared adapter is slightly smaller at effectively identical quality, while FiLM provides better quality for only 380 additional bytes. The tested Mirror therefore does not improve the storage/quality frontier over the cheapest ordinary control. The full-model storage gain is negligible.

### Counter-hypothesis (C)

The box prompts and generated shapes make zero-shot SAM nearly saturated (0.9937 mean mIoU), leaving little useful task variation to recover. Also, final-embedding coordinates may not represent the layer-specific mixture of LoRA, prefix, and Adapter paths used by MoPEFT. These results support neither a general MoPEFT failure nor a general Mirror failure.

### Unknown (U)

Natural-image domains, MESS, official internal-layer MoPEFT, larger audit sets, training with image-encoder fine-tuning, other SAM backbones, object-level tasks, and GPU throughput remain untested. The synthetic benchmark is a small mechanism screen, not capacity evidence.

## Reproduction and verification

Development settings and the development manifest were committed at `4ed0515c3` before audit generation. Audit images/masks/manifests, prediction files, serialized payloads and row records are retained under `runs/`. `source/verify_result.py` replays the saved metrics, verifies each prediction/payload hash, and checks full-deployment byte arithmetic. Six tests pass. The exact SAM checkpoint is cached outside Git; its SHA is recorded above and in `VERIFICATION.json`.
