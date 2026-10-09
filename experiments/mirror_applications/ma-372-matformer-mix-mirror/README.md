# MA-372 — MatFormer Mix'n'Match factorized Mirror codes

Status: **FAIL for Mirror-specific value**  
Branch: `research/ma-372-matformer-mix-mirror-20261009`  
Base: `0eded323`  
Prior art: PA53 MatFormer

## H

Factorized layer × granularity Mirror coordinates recover untrained mixed-width configurations with fewer bytes than a flat configuration table while retaining quality; Mirror should also beat an ordinary factorized coefficient table to establish specificity.

## Frozen mechanism screen

Four FFN layer roles and three nested widths form a 4×3 function bank. Twelve target scalar channel-gain operators are generated compositionally from a shared base, a layer factor, and a granularity factor. Six combinations are observed for fitting; six are withheld and evaluated. Compare native width-only shared-prefix behavior, flat per-configuration gains, factorized Mirror, and direct factorized coefficients. Two development seeds; actual deterministic NPZ bytes include all weights, codes and metadata. This is an oracle composition screen, not MatFormer training.

PASS requires exact held-out function recovery, at least 10% fewer bytes than flat storage, and at least 10% byte advantage over direct factorized coefficients at equal quality. FAIL if direct coefficients match Mirror function/bytes or held-out function recovery fails. Fresh sealed on direct-control alias.

## H / T / D / C / U

- **H:** Factorized Mirror codes recover unseen layer×width mixes more cheaply than flat storage and native Mix'n'Match.
- **T:** Frozen 4×3 linear operator bank, two development seeds, six held-out combinations, 8 payload/metric rows, no optimizer updates. Fresh sealed.
- **D:** FAIL for Mirror-specific value: factorized Mirror recovers held-out compositions exactly at 1401 B vs 6017 B flat; direct factorized coefficients match exactly at 1401 B. Native width-only is 737 B but nMSE .10498 and collapses to one distinct function.
- **C:** This is ordinary low-rank matrix factorization; native MatFormer already exposes mix-and-match widths.
- **U:** Actual MatFormer checkpoint training, transformer NLL, GPU throughput, task-level capacity.
