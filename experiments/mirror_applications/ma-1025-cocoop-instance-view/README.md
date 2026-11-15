# MA-1025 — CoCoOp image-conditioned Mirror prompt code

Status: FAIL (scoped CPU screen)  
Evidence lane: LANGUAGE / STORAGE / COMPUTE  
Base commit: `5381123b3` (MA-1024 evidence tip)  
Prior art: PA313 (CoCoOp), with PA312 CoOp baseline.

## Hypothesis

H: A two-value image-conditioned Givens code can preserve CoCoOp base-to-novel accuracy while using fewer generator bytes, and improve on a matched two-value additive low-rank prompt generator.

## Mirror insertion

> **Mirror insertion:** this experiment adds per-image code `m(x)∈R²` to a shared four-token CLIP context by Givens rotations so that each image gets a logical conditional text prompt without storing a full generated context per image.

- Shared object: frozen CLIP ViT-B/32, one four-token context, and a shared image conditioner.
- Mirror coordinate: two values generated from the frozen CLIP image feature and applied as paired prompt-token rotations.
- Controls: zero-shot CLIP, shared CoOp, full CoCoOp-style 512D generated prompt bias, and a same-width rank-2 additive residual generator.
- Data: CIFAR-10 base classes 0–5 for training and novel classes 6–9 for audit. Image indices are disjoint from MA-1024; the official test split opens only after model freeze.

## Protocol

See `PROTOCOL.json`. Four training images per base class, 20 fixed updates, minibatch 4. CoCoOp and Mirror generate a per-image text prompt at inference; text-tower latency and all generator/context/backbone bytes are charged.

## Gates

- **PASS:** Mirror novel accuracy is within 1 pp of CoCoOp, prompt generator bytes are at least 25% lower, and Mirror beats the rank-2 additive control by at least 1 pp at matched updates.
- **FAIL:** rank-2 additive control matches/beats Mirror, or Mirror misses the CoCoOp quality gate.
- **NOT ESTABLISHED:** audit index overlap, asset/replay failure, or prompt state is not fully reconstructible.

CIFAR-10/CLIP is a bounded screen, not broad VLM evidence. MA-1025's 24 training and 40 official-test images are index-disjoint from MA-1024. All four learned methods received 20 updates at batch size 4. Audit metrics were computed only after all learned states were frozen.

## Result

Mirror reached 81.25% novel accuracy (13/16), compared with 75.00% (12/16) for the full CoCoOp-style control. It used 74,935 learned-state bytes versus 142,236 B for CoCoOp. However, the matched rank-2 additive control also scored 81.25% novel and 83.33% base, exactly matching Mirror, and used 91,390 B. This misses the preregistered requirement that Mirror exceed the additive control by at least 1 pp. The result is **FAIL for Mirror-specific value**.

The complete frozen-CLIP bundle is 608,938,515 B for Mirror versus 608,954,970 B for rank-2 additive and 609,005,816 B for CoCoOp. Against the 608,863,580 B backbone, the Mirror delta is 74,935 B (0.0123%). The modest full-bundle changes do not constitute meaningful end-to-end model compression.

See `RESULTS.md` for the H/T/D/C/U record, exact timings, serialized replay, and limits.
