# MA-830 — task × layer KanAdapter codes

## H / T / D / C / U

**H:** Separate task and layer Views preserve heldout task-layer adapter functions with lower complete bytes than independent KanAdapter banks and beat native HyperFormer/GS-KAN factors.

**T:** Synthetic analytic two-function basis (`sin(x)` and `x^3`); eight tasks × six layers; checkerboard 24/24 pair split; 24,576 heldout values; seed 83001; zero optimizer updates. Compared independent sampled pair functions, native task/layer factors, Mirror task/layer Views, task-only and layer-only controls.

**D: FAIL for Mirror-specific attribution.** Mirror uses 1,253 B and yields zero heldout error, while the native factor control uses 1,233 B and also has zero error. Independent 64-point pair tables use 11,277 B and have nRMSE .000545 from interpolation. Task-only and layer-only controls fail badly. Mirror and native factors use the same number of parent evaluations; the native payload is 20 B smaller.

**Evaluation boundary:** Heldout task/layer IDs and both factor tables are stored in the payload; this tests decoding, not inference of unseen tasks or layers.

**C:** Native task/layer hypernetwork-style factorization explains the behavior with fewer bytes.

**U:** Trained KanAdapter/HyperFormer, real downstream task quality, learned code inference, continual retention, and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Complete payload bytes, hashes, errors, compute proxy and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** The factorization compresses a synthetic pair bank, but offers no Mirror-specific advantage over the native factorization.
- **Hypothesis:** A useful result would need nonseparable heldout interactions where a Mirror code beats a generated adapter or private residual at matched bytes.
