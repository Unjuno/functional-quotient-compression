# MA-1162 — single-learned-scalar causal Mirror pilot (2026-10-18)

**Status: FAIL to learn delayed copy under this particular frozen random rule; mechanism/causality check PASS. Not evidence against general Mirror-native networks.**

Independent exploratory follow-up to MA-1162 L1. The separate container-run Python source and CSV are supplied in the research conversation and have *not* been incorporated into this repository by this report.

## Exact scope
- One **trainable scalar** theta, reused as the relative Q/K Givens rotation and nonlinear V sandwich code, one causal attention residual block.
- All embedding/QKV/output physical tensors remain **fixed seeded FP32 buffers**, NOT trainable parameters. This is NOT one total stored numerical value, NOT a trained full Transformer, and NOT a fair byte-matched competitor to MA-1162 L1.
- d=16; vocabulary 32; length 12; query final symbol 31 and label is token at position -2, with 31 target classes.
- CPU PyTorch 2.10.0+cpu eager, 1 thread; 3 independent seeds 0/1/2, 120 Adam updates, batch 64, LR 0.04. Test 2048 held-out synthetic examples/seed.
- Three 1-scalar variants: Mirror, common-QK null gauge, gain. Additional zero-trained-scalar frozen control.
- Serialized full inference payload: 8,853 B for trainable variants, 8,789 B for frozen. Stored scalar tensors: 1,537, learned scalars: 1 (0 for frozen). Optimizer excluded.

## Observations (mean across 3 seeds)

| Variant | Test NLL (nats/query) | Accuracy | Median training wall (s) |
| --- | ---: | ---: | ---: |
| Mirror | 3.468665 | 0.016276 | 0.18420 |
| Common QK gauge | 3.485569 | 0.009766 | 0.14373 |
| Gain | 3.483879 | 0.009277 | 0.12587 |
| Frozen | 3.485569 | 0.009766 | 0.04973 |

Chance accuracy for the actual 31-label target is ~1/31. Accuracy below chance is possible under the fixed, biased output projection. Mirror theta changed in every seed, but accuracy did not approach adequate retrieval performance. Causal prefix perturbation was exactly 0. Five independent tests passed for one trainable scalar, gauge invariance, nonzero finite gradient, causality, and serialized checkpoint equality.

## Source/evidence integrity

Container output SHA-256:
- `experiment.py`: `2c7c9456b30166eda6bb669d5515bb038921cbbd05d2270f61e7c98342ba727b`
- `test_experiment.py`: `ac9de99a94cba264eea57ece864f11e497445659569bceccb2ef6a7f87feb11b`
- `results.csv`: `0bed4f45845da7a46c79c8a4efacc0ee2422b0e25824915ed4544270bdb62e1c`

## Interpretation
Single-scalar optimization cannot in general redesign arbitrary frozen lexical embeddings and output/attention matrices. A small NLL change with no meaningful accuracy is **not success**. This is a diagnostic one-block causal Transformer-style mixer, not a multi-layer, fully learned Transformer. Next research decision should distinguish **one shared Mirror address with learned common core** versus **one learnable scalar in the entire inference system**. Both require same-byte and same-compute controls before adoption. No MA-1162 registry status changes and no prior FAIL is overwritten.
