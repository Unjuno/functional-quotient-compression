# MA-356 — Product-key factorized Mirror code composition

Status: **FAIL for Mirror-specific value; narrow oracle factorized-code result**  
Branch: `research/ma-356-factorized-mirror-code-composition-20261009`
Base: `research/mirror-application-worker-ready-20261007` (`c935a90`)  
Prior art: PA43 Product Key Memory.

## H — Hypothesis

A Cartesian product of two compact Mirror subcodes can represent held-out combinations of functional factors with fewer actual bytes than a flat function-code table while retaining quality and useful function diversity.

## Mirror insertion

> **Mirror insertion:** this experiment adds a product of two categorical subcodes to one shared operator basis so each pair composes a different logical linear function without storing a complete matrix per pair.

## T — Test

Oracle synthetic 8×8 linear operators. Each target is `W0 + alpha_i U + beta_j V + alpha_i beta_j C`, with eight values per coordinate and 64 pair functions. Development seeds 35601 and 35602. Forty-eight pairs were marked observed and 16 held out; all pair maps were scored using 256 fixed Gaussian probes. No training updates: this is a post-fit representation mechanism screen. Actual deterministic ZIP/NPY payloads were reloaded before scoring.

Controls: flat 64-function matrix table, Product-Key keys plus explicit value table, factorized Mirror code, direct coefficient code using the same basis. Fresh seeds 35611–35613 remained sealed.

## D — Decision

**FAIL for Mirror-specific value.** On both seeds the factorized code recovered all 64 functions, including held-out pairs, at zero nMSE and zero collisions. Payload was 2,628–2,631B versus 15,523–15,538B flat and 15,900–15,915B Product Key. Lookup proxy was 2 factor reads versus 64 full-value accesses. The direct coefficient control used 2,629–2,632B, only 1B larger at equal quality, failing the preregistered 10% Mirror-specific margin.

## Fact / Interpretation / Hypothesis

**Fact:** All 64 outputs were distinct and exactly reconstructed. Factorized Mirror state was about 2.63KB; direct coefficients were byte-near at +1B. Flat/Product-Key tables were about 15.5–15.9KB.

**Interpretation:** The measured compression follows from representing the bank with three shared basis matrices and two coordinates; product composition makes 64 logical functions from this state, but the direct coefficient control does the same at effectively equal cost. The combinations do not establish independent capacity.

**Hypothesis:** Learning a shared basis from partial observed pairs may make compositional codes useful, but would require a new protocol that charges factor fitting and tests unseen pair identities.

## C — Strongest counter-hypothesis

The task is constructed to match a low-rank bilinear factorization exactly, and all basis values are oracle-provided. The result does not establish learned compositionality or generalization.

## U — Unconfirmed

No end-to-end codebook learning, noisy interactions, natural data, training cost, or fresh replication.
