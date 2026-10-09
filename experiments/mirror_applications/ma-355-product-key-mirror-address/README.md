# MA-355 — Product-Key Mirror address lookup

Status: **FAIL for Mirror-specific gate; PROMISING only as an oracle factorized-basis storage point; fresh sealed**  
Branch: `research/ma-355-product-key-mirror-address-20261009`
Base: `research/mirror-application-worker-ready-20261007` (`c935a90`)  
Prior art: PA43 Product Key Memory.

## H — Hypothesis

A factorized Mirror address can select distinct functions across a 16×16 Cartesian bank with lower total bytes and routing work than Product Key or a flat matrix table, while preserving held-out-composition quality.

## Mirror insertion

> **Mirror insertion:** this experiment adds a pair of factorized addresses `(m1,m2)` to one shared linear-map basis so a Cartesian pair selects one logical operator without storing a full matrix for every pair.

## T — Test

Synthetic 8→4 linear bank, 16 left factors × 16 right factors. The initial development attempt included a full 256-entry interaction tensor in factorized methods and scored held-out pairs against that same stored tensor. This invalid run was excluded and documented in PROTOCOL.json before any fresh access. Amended protocol set interaction exactly to zero; pair outputs are reconstructed from base+left+right factors only. Two development worlds (35501/35502); 192 pair IDs designated seen and 64 held out. No training updates; factor values are oracle supplied. Actual ZIP/NPY serialization includes full state, indices, and metadata.

Controls: independent and flat full matrices, ordinary Product-Key keys plus full value table, factorized Mirror address, ordinary direct two-index coefficient control. Fresh seeds 35511–35513 remained sealed.

## D — Decision

**FAIL for Mirror-specific value.** Amended Mirror factorization exactly recovered all 256 additive maps with zero held-out nMSE and zero collisions at 5,128–5,133B, versus 30,578–30,620B independent/flat and 30,982–31,020B ordinary Product Key. The direct two-index coefficient control was only 12B larger (5,140–5,145B), below the preregistered 10% margin. Routing proxy for factorized direct lookup was two index reads versus 256 MACs for Product-Key retrieval. This supports a narrow oracle factorized-basis storage result, not a Mirror-specific gain or trained routing result.

## Fact / Interpretation / Hypothesis

**Fact:** All amended methods had zero measured reconstruction error on this planted additive bank. The factorized payload used about 5.1KB; flat and Product-Key controls used about 30.6–31.0KB. The direct coefficient control was 12B larger. Each method yielded 256 distinct maps.

**Interpretation:** Most storage reduction came from the known additive factorization, which is expressible with ordinary direct indices and factors. The factor count is not independent capacity; all 256 functions are known oracle combinations.

**Hypothesis:** A learned factorization under interaction noise might expose when the address code is useful, but requires a separate protocol with training of shared factors and held-out pair identities.

## C — Strongest counter-hypothesis

This is an oracle additive task constructed to fit exactly in a factorized basis. It demonstrates only representation/storage economics under exact structure, and the direct control defeats Mirror specificity.

## U — Unconfirmed

No learned key/router, interaction robustness, natural data, generalization beyond the supplied factors, or fresh replication.
