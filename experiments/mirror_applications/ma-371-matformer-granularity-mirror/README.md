# MA-371 — MatFormer granularity Mirror views

Status: **FAIL for Mirror-specific value**
Branch: `research/ma-371-matformer-granularity-mirror-20261009`
Base: `bccbc91a`
Prior art: PA53 MatFormer

## H — Hypothesis

Per-layer, per-granularity Mirror gains on a shared nested FFN can recover useful behavior for unseen mixed-width Transformer configurations, beyond native nested-width selection, at lower actual bytes than independently stored width-specific FFNs and with an advantage over ordinary per-layer gains.

## Insertion

Two-layer residual FFN network with one shared maximum-width W1/W2 pair per layer. Granularity selects a nested prefix; Mirror code applies diagonal channel gains to the selected hidden activation. A matched ordinary direct-coefficient control stores the same gains. Controls include native shared-prefix network, direct gains, Mirror gains, independently trained width-specific FFNs, and held-out mixed granularities.

## Frozen protocol

Synthetic teacher provides three width roles per layer and a 3×3 fixed design of layer granularities; train only the six homogeneous granularity configurations. Evaluate the three off-diagonal configurations as untrained mixes on the same 4-class task. Two development seeds, fixed train/test worlds, 1,000 SGD updates. Actual NPZ inference payload includes shared weights, per-granularity codes and metadata. Report NLL, accuracy, unseen-mix retention, bytes, matrix-MAC proxy and training wall time. No fresh split is opened if the exact direct-gain alias gate fails.

## Gates

PASS: Mirror reaches mean unseen-mix accuracy within 2 percentage points of independent per-configuration models, costs at least 10% fewer bytes than the independent bank, and beats direct coefficients by at least 10% on bytes at equal quality. FAIL if Mirror and direct coefficients have equivalent function/payload or if no unseen-mix quality benefit over native nested selection.

## H / T / D / C / U

- **H:** Granularity-specific Mirror views recover useful untrained MatFormer mixes at less storage than independent FFNs and improve the storage/quality frontier over native gains.
- **T:** Two development seeds, 2-layer nested FFN, widths 2/4/8; three trained homogeneous configs and three held-out mixes; 48 rows. Initial unlearnable task was amended before adjudication; fresh sealed.
- **D:** FAIL for Mirror-specific value: direct coefficients equal Mirror in representation and bytes (2084 B/world); shared prefix 1670 B and independent bank 3942 B. Held-out mix gains are inconsistent.
- **C:** PA53's native nested width selection or a simple diagonal per-layer gain may explain any gain.
- **U:** Actual MatFormer checkpoints, language-model perplexity, long training, GPU throughput, hardware latency.
