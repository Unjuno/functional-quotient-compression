# MA-1024 — CoOp prompt bank replaced by structured Mirror code

Status: SCREENING
Evidence lane: LANGUAGE / STORAGE / COMPUTE
Base commit: `64d11ac4c` (quantum family diagnostic branch tip)
Prior art: PA312 (CoOp), with PA313 as a nearby conditional-prompt control.

## Hypothesis

H: A small domain phase code acting on a shared four-token CoOp context can retain base-to-novel CIFAR-10 classification quality across four visual domains while using substantially fewer prompt bytes than independent domain CoOp contexts.

## Mirror insertion

> **Mirror insertion:** this experiment adds one domain coordinate `m_d` as paired Givens rotations on a shared four-token CLIP text context so that domain-specific logical classifiers can reuse one physical prompt basis instead of storing one full context per domain.

- Shared physical object: frozen OpenAI CLIP ViT-B/32 and one four-token text context basis.
- Mirror coordinate: one scalar angle per image domain, rotating two pairs of context-token vectors.
- Simpler control: ordinary native scalar Givens phase on the same context basis; this is a matched functional control.
- Upper control: independent four-token CoOp context per domain.
- Data: CIFAR-10 base classes 0–5 for context learning; official test split novel classes 6–9 reserved for fresh audit. Domains are original, grayscale, reduced brightness and Gaussian blur.

## Protocol

See `PROTOCOL.json`. Use 4 labeled examples per base class, 30 fixed optimizer updates, and no audit-driven tuning. The frozen CLIP weights, tokenizer and processor are included in complete serialized inference payload accounting; prompt-specific payload is also reported separately.

## Gates

- **PASS:** Mirror's base/novel harmonic mean is within 1 percentage point of independent-domain CoOp, actual prompt bytes are at least 25% lower, and it exceeds shared CoOp by at least 2 points on novel classes.
- **FAIL for Mirror-specific value:** native scalar Givens control matches Mirror within 0.1 point, or the quality/storage gate is missed.
- **NOT ESTABLISHED:** data/model assets, exact payload replay or the frozen split audit fails.

This is a small CIFAR-10/CLIP screen, not broad VLM generalization evidence.
