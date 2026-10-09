# MA-392 — Token × domain factorized embedding Mirror address

Status: protocol frozen before development. Dedicated branch: `research/ma-392-token-domain-mirror-embedding-20261009`.

## Mirror insertion

**Mirror insertion:** apply a small domain-specific Givens coordinate to a shared token embedding, producing many logical domain-token vectors without storing one vector per pair.

## H — Hypothesis

For domain-conditioned embeddings formed by orthogonal transformations of a shared token table, domain Mirror angles preserve unseen token-domain combinations while reducing complete payload bytes relative to independent tables and dense domain transforms, and improve over hard sharing and FiLM.

## T — Frozen design

PA59 motivates factorized addresses. The experiment has 128 tokens, eight domains, 16 dimensions, a fixed shared token table, and an 80/20 token-domain split in two development worlds. Compare hard sharing, FiLM, dense domain maps and eight domain-level Givens angles. A full independent oracle stores all outputs for storage/quality reference only; it is not used for held-out generalization claims. All shared tables, views, decoder and archive metadata are charged. See `PROTOCOL.json` for frozen details and gates.

Teacher domains apply disjoint 2D rotations to token vectors. This aligned synthetic screen isolates a domain-level functional coordinate; it does not test natural recommender or language data.

## D — Result

**Fact.** Seeds 39201/39202 had 819/815 observed pairs and 205/209 held-out pairs. Mirror held-out NRMSE was 1.41e-7/3.33e-8, matching the independent oracle, and exceeded both FiLM (.392/.427) and hard sharing (.422/.449). It also beat the dense domain transform (.0162/.0143). Mirror used 10,291/10,275B versus 17,047/17,041B dense (60.4%/60.3%; frozen limit ≤50%) and 63,024/63,083B for the full oracle. All ten actual payloads replayed exact bytes, hashes and metrics; four tests pass. Fresh 39211–39213 remained sealed.

**Interpretation.** The domain-level Givens coordinate generalized across unobserved token-domain pairs in this aligned teacher while using 39.6–39.7% fewer bytes than generic dense domain transforms. It meets the held-out fidelity and simple-control quality margins, but misses the preregistered 50% dense-payload target because the shared token table is a large common cost. Mirror transform proxy is 32 MACs + 16 adds + 16 trigonometric operations per pair versus dense 256 MACs; measured CPU throughput was ~6.9–7.8M pairs/s for Mirror versus ~3.1–3.4M for dense and ~7.7–8.9M for FiLM. The strict result is FAIL on total storage.

**Hypothesis.** A low-description domain transform captures the true shared-domain structure; byte dilution from the fixed token table dominates at eight domains and 128 tokens. Scaling domain count could improve the ratio, but that is a new experiment and is not established here.

## H / T / D / C / U

- **H:** Domain Givens angles generalize to unseen token-domain pairs and compress against dense maps while beating hard sharing and FiLM.
- **T:** PA59; 128 tokens × 8 domains × 16 dimensions; fixed token table; 80/20 pair split; seeds 39201/39202; independent oracle, hard sharing, FiLM, dense transforms and Mirror; 2,000 Adam updates for trainable methods.
- **D:** **FAIL at frozen complete-payload gate.** Mirror held-out NRMSE was near zero and beat dense, FiLM and sharing, but payload remained 60.3–60.4% of dense versus required ≤50%; fresh stayed sealed.
- **C:** Dense transforms also generalize well and save 39.6% versus full independent vectors; common table bytes dilute the Mirror code advantage.
- **U:** Fresh worlds, real multi-domain vocabulary/recommender data, learned shared tables, larger domain scaling, and private token-domain residuals.
