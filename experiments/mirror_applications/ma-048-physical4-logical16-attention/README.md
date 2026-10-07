# MA-048 — physical-4 to logical-16 attention heads

Status: SCREENING. Branch `research/ma-048-physical4-logical16-attention-20261007`.

## H

Four physical QKV sets expanded to sixteen logical heads with per-head Givens views may recover an aligned 16-head teacher at lower serialized payload than ordinary MHA. Hard GQA and rank-1 generated residuals are controls.

## T

Synthetic 32D full-attention block, sequence length 6, sixteen 2D heads, four physical QKV groups. Compared full MHA, hard GQA, rank-1 per-logical-head QKV residual, and Mirror views. A second teacher uses sixteen independent QKV sets. Development world 48000 selected LR 0.003; fresh worlds 48001–48003 remain sealed. Hash manifest records protocol, source, tests, selection and development data.

## Development screen

Aligned MSE at selected LR 0.003: full MHA 0.01557, Mirror 0.02047 (1.315x), rank-1 residual 0.01207, hard GQA 0.03529. Serialized payload: 10,401B Mirror vs 18,405B full MHA and 16,093B rank-1. Independent16 mode remained difficult for shared methods. Fresh worlds remain unopened.

## Decision

FACT: fresh verification pending. INTERPRETATION: storage-quality tradeoff is possible but selected development MSE missed the strict ratio. HYPOTHESIS: fresh results may vary. BOUNDARY: synthetic attention only.
