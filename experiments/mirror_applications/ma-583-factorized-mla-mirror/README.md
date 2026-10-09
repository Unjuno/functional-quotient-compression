# MA-583 — Factorized head × layer MLA Mirror code

Status: **FAIL for Mirror-specific value**  
Branch: `research/ma-583-factorized-mla-mirror-20261009`  
Base commit: `0d4c52ff`  
Prior art: PA115 Multi-Head Latent Attention

## H

Factorized head×layer Mirror coordinates reconstruct held-out MLA key/value role maps from one shared latent basis with fewer bytes than per-role tables, without collapsing useful role functions; Mirror should beat direct coefficients to be specific.

## Frozen mechanism screen

Four heads × three layers, 8D latent to 4D KV maps. Target operators use a shared base and a rank-one head×layer interaction. Six of twelve roles are observed; six are held out. Compare native MLA repeated shared map, flat role table, factorized Mirror, and direct factorized coefficients. Two dev seeds; actual NPZ bytes, held-out map nMSE, distinct roles, reconstruction MACs and wall time. Oracle algebra screen, zero optimizer updates.

PASS: held-out nMSE ≤1e-6, all six held-out roles distinct, ≥20% fewer bytes than flat, and ≥10% fewer than direct coefficients. FAIL on direct alias or role collapse. Fresh sealed.

## H / T / D / C / U

- **H:** Small head×layer coordinates preserve many useful roles from one MLA latent basis.
- **T:** Four heads × three layers, six held-out roles, two seeds; eight rows, no updates.
- **D:** FAIL for Mirror-specific value. Factorized Mirror/direct both 1437B and exact; flat table 4389B. Native MLA is 600B but collapses to one role (nMSE .202).
- **C:** Native MLA and ordinary factorized coefficients.
- **U:** Attention-weighted quality, real MLA checkpoint, GPU cache footprint and throughput.
