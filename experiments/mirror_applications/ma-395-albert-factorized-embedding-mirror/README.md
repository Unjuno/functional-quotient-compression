# MA-395 — ALBERT factorized embedding with latent-domain Mirror views

Status: protocol frozen before development. Dedicated branch: `research/ma-395-albert-factorized-embedding-mirror-20261009`.

## Mirror insertion

**Mirror insertion:** add four per-domain Givens angles inside the shared 8-dimensional ALBERT-style embedding bottleneck, before its common projection to 16 dimensions.

## H — Hypothesis

A low-dimensional latent Mirror view can recover held-out domain-token embedding functions at lower actual bytes than dense domain adapters, while improving over hard factorized sharing and output-side FiLM.

## T — Frozen design

PA61 ALBERT factorized embeddings provide the baseline. A fixed shared 128×8 table and 8×16 projection feed eight domain functions; a seeded 80/20 pair split tests held-out combinations. Compare independent-table oracle, hard factorized sharing, output FiLM, dense 8×8 latent transforms, and four-angle latent Mirror views. Every table, adapter, decoder and metadata object is charged. See `PROTOCOL.json` for frozen gates and seeds.

The synthetic teacher uses latent Givens rotations by construction. This isolates whether the coordinate is useful inside the bottleneck; it is not pretrained ALBERT or natural transfer evidence.

## D — Result

**Fact.** Across seeds 39501/39502 (820/204 and 854/170 observed/held-out pairs), Mirror held-out NRMSE was 0/3.1e-8, compared with dense latent transforms .0152/.00310, FiLM .3159/.3015 and hard sharing .3214/.2991. Mirror payload was 7,293/7,268B versus dense 8,975/8,942B (81.3%/81.3%, above the frozen ≤80% limit), and the full oracle was 63,388/63,307B. Mirror and dense both matched full decoder NLL/top-1. All ten payloads replayed exact sizes, hashes and metrics; four tests pass. Fresh 39511–39513 remained sealed.

**Interpretation.** In this aligned synthetic fixture the latent view recovers all held-out domain outputs while using 18.7% fewer complete bytes than dense latent maps and about 88.5% fewer than independent tables. It improves greatly over hard sharing and output-side FiLM. The common base table, projection and decoder dominate the complete payload, leaving the result just short of the frozen dense-comparator target. The Mirror transform uses 16 latent MACs, eight adds and eight trigonometric operations per pair, versus 64 transform MACs for dense maps; observed CPU throughput was about 9.2–9.5M pairs/s for Mirror versus 7.9–8.0M dense and 11.5M FiLM.

**Hypothesis.** Moving the view inside the bottleneck provides a more efficient functional coordinate than an unrestricted dense latent matrix, but complete storage savings are diluted by shared factorization state. No natural ALBERT transfer result is established.

## H / T / D / C / U

- **H:** Four domain Givens angles inside the 8D factorized embedding bottleneck recover unseen domain-token functions more compactly than dense latent adapters and outperform hard sharing/FiLM.
- **T:** PA61; 128 tokens × 8 domains, fixed 128×8 base and 8×16 projection, 80/20 split, seeds 39501/39502, five controls, 2,000 Adam updates for trainable controls, actual compressed NPZ payloads.
- **D:** **FAIL at the strict complete-payload gate, with excellent held-out functional recovery.** Mirror NRMSE was essentially zero and improved on all shared baselines, but total bytes were 81.3% of dense versus the frozen ≤80% requirement; fresh remained sealed.
- **C:** Generic dense transforms also generalized and only used about 23.5% more bytes; fixed-table sharing makes their byte gap smaller than the parameter-count gap suggests.
- **U:** Fresh worlds, pretrained ALBERT or natural vocabulary transfer, learned factorization state, larger domain counts, quantization and private residuals.
