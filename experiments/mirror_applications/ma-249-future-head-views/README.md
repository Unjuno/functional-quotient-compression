# MA-249 — one physical future head + Mirror future-offset views

Status: **PROMISING (aligned synthetic head-sharing mechanism)**  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `4fe9b5fe51bfcd488ddff9a9702f8cb9bee4b08a`

## H — falsifiable hypothesis

When future-offset teachers share one output matrix under low-dimensional hidden-space Givens transforms, one physical MTP head plus offset-specific Mirror coordinates can recover independent-head quality with lower actual serialized model bytes. On unaligned offset heads it should fail or require private residuals.

## T — execution

PA09 (Multi-token Prediction) uses a shared trunk with separate future-token heads. We compared ordinary four-head MTP, hard-tied output head, scalar-gated shared head, tied head plus rank-1 residual, one shared head plus Givens Mirror views, and ordinary MTP as the quality upper control.

A common frozen orthogonal feature map transforms 16D synthetic inputs. Four future offsets each predict a categorical distribution over 12 outputs. The aligned teacher uses one shared output matrix viewed through offset-specific Givens rotations. The negative control uses independently sampled output matrices per offset. Models were trained by soft-target distillation for 1,200 AdamW updates, batch 64, one CPU thread. Dev world 24900 selected common LR 0.01. Fresh worlds were 24901–24903. Actual payloads include the shared feature map, method tensors, and serialized config.

## D — PROMISING on the aligned mechanism; independent-offset control fails

### Aligned shared-base teacher

| Fresh world | Ordinary MTP KL / top-1 | Mirror KL / top-1 | Mirror payload |
|---|---:|---:|---:|
| 24901 | 2.71e-8 / 100% | 1.65e-9 / 100% | 4,384 B |
| 24902 | 4.41e-9 / 100% | 2.49e-9 / 100% | 4,384 B |
| 24903 | 3.73e-9 / 100% | 1.89e-9 / 100% | 4,384 B |

Mirror passed the registered aligned gates in 3/3 fresh worlds. Ordinary MTP used 6,497 B, so Mirror used 32.5% fewer actual serialized bytes. Hard tying (4,066 B) and scalar gating (4,325 B) were smaller but reached only 65–71% top-1 agreement. Rank-1 private residual used 4,893 B and reached 73–79%; Mirror was both smaller and more accurate. Mirror was 59 B larger than scalar gating, but had much better quality.

### Independent offset matrices

Ordinary MTP remained exact. Mirror top-1 agreement was 37.6–38.6% with KL 0.730–0.743. Rank-1 private residual improved this to 40.4–43.2% and KL 0.658–0.682, while using 509 B more than Mirror. This is evidence that private offset degrees of freedom help when the head matrices are not related by the registered view family; rank 1 still did not recover the independent-head upper control.

### Storage / compute / runtime

Aligned actual payload: ordinary MTP 6,497 B; Mirror 4,384 B (-32.5%); tied 4,066 B; scalar gate 4,325 B; rank-1 residual 4,893 B. The shared feature map is included in every payload. No object is free.

Median aligned training wall time was 1.15 s for MTP and 1.53 s for Mirror (+34%). The active compute proxy was 176.9M vs 191.7M (+8.3%). Median CPU throughput was 1.44M examples/s for MTP and 0.604M for Mirror (0.42x). This tiny eager benchmark suggests the view transform costs runtime; optimized kernels and accelerator performance are untested.

## C — strongest counter-hypothesis

The aligned teacher is deliberately generated from the same Givens-view family tested by Mirror, making it an optimistic recovery case. This establishes a mechanism/storage result only. Also, the apparent CPU throughput regression may come from the unoptimized eager coordinate transform and does not establish a hardware-level lower bound.

## U — not established

Natural-language MTP quality, actual speculative accepted-token rate, long horizons, learned/unfrozen trunks, broader view families, near-convergence fixed-byte capacity, and optimized kernels remain untested. The independent-offset mode only shows that this Givens parameterization is insufficient for arbitrary per-offset matrices; it does not identify the minimum private rank.

## Fact / interpretation / hypothesis

- **Fact:** Aligned Mirror matched the MTP top-1 distributions in all 3 fresh worlds, used 32.5% fewer serialized bytes, and ran slower in the current CPU implementation. On independent heads it was far from MTP, while rank-1 private residual improved quality.
- **Interpretation:** A low-description Givens coordinate can replace several future heads when those heads share the corresponding hidden-space relation. This trades quality-neutral storage savings for additional compute/runtime and fails outside the aligned family.
- **Hypothesis:** An optimized fused rotation kernel may recover throughput while preserving the byte frontier; a follow-up would need a separate protocol and real MTP context features.

## Verification

- Tests: `python -m pytest -q experiments/mirror_applications/ma-249-future-head-views/tests` (4 passed).
- Fresh metric replay: 30/30 rows; maximum KL delta 4.92e-11; maximum top-1 delta 4.38e-9; all serialized payload byte counts matched exactly. See `VERIFICATION_REPLAY.json`.
- The frozen protocol/source hashes and pre-fresh stage are in `FREEZE_MANIFEST.json`.
