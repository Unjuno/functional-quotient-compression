# MA-930 — factorized model-stitching bridges

Status: SCREENING

Frozen amendment 01 uses three public TinyStories GPT-Neo language models with the same tokenizer, 8 layers and hidden widths 64/128/256. It tests a source × target × layer factorization on paired hidden states, holding out both directions between the smallest and largest models. The strong control is a full independent affine bridge for each pair/layer trained on the same fitting stories.

This is a real-LM representation-transfer and storage screen, not StitchLLM reproduction, semantic-equivalence evidence, or end-to-end serving throughput. It includes held-out target-tail logit drift and next-token NLL delta to expose losses hidden by activation fit. Protocol frozen before data/model download.
