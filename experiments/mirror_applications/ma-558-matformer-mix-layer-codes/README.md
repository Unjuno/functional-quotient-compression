# MA-558 — MatFormer mix-and-match Mirror layer codes

Status: **FAIL for Mirror-specific value**  
Branch: `research/ma-558-matformer-mix-layer-codes-20261009`  
Base commit: `ae62f3aa`  
Prior art: PA107 MatFormer

## H

Layer-indexed Mirror granularity codes improve held-out mixed-width submodels beyond native nested FFN selection while storing less than independent per-configuration models and direct coefficient controls.

## Frozen mechanism screen

Three FFN layers each choose width 2/4/8, yielding 27 submodels. Six homogeneous and selected adjacent mixes are treated as observed; six locked alternating mixes are held out. Synthetic targets follow a shared base plus layer×width low-rank interaction. Compare native prefixes, flat held-out target table, factorized Mirror code and direct factorized coefficients. Two seeds; actual NPZ bytes and held-out operator nMSE, distinct functions, MAC proxy and wall time. No optimizer updates; oracle aligned screen only.

PASS: held-out nMSE ≤1e-6, ≥10% smaller than flat, and ≥10% smaller than direct coefficient at equal quality. FAIL on direct alias. Fresh sealed.

## H / T / D / C / U

- **H:** Layer-wise Mirror codes improve untrained MatFormer mixtures.
- **T:** Three layers × widths 2/4/8, two seeds, six held-out mixed configurations; eight rows, zero optimizer updates.
- **D:** FAIL for Mirror-specific value. Factorized Mirror exactly reconstructs all held-out mixes at 1364B, but direct coefficients match exactly; flat table 27564B and native prefix 728B with nMSE .341–.798.
- **C:** Native MatFormer width mixing and ordinary factorized coefficients.
- **U:** Actual pretrained MatFormer, language quality, serving throughput.
