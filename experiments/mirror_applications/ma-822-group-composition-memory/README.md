# MA-822 — group-composition matrix memory

## H / T / D / C / U

**H:** A Mirror group-element code over a shared cyclic generator reaches faithful matrix composition at lower full bytes and compute than independent matrices and native group indices.

**T:** Exact C16 rotation representation in 2D; 16 elements and 32 heldout pair compositions; no training; controls are independent matrix table, Mirror generator+codes and native cyclic group index.

**D: FAIL.** Independent matrix table uses 574 B and 2 composition operations in the proxy. Mirror code uses 858 B; native cyclic index uses 854 B; both have nRMSE 3.08e-7 and 4 operation proxy. The native group code is 4 B smaller with identical function/stability. Both compressed code payloads are larger and more compute-heavy than the tiny independent matrix table at this size.

**C:** The standard cyclic group law (integer index modulo 16 plus matrix power) exactly explains the Mirror code; metadata and generator reconstruction are not economical at this small group order.

**U:** Learned group discovery, larger groups/dimensions, neural memories, natural composition tasks and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Payload sizes/hashes, composition error, orthogonality, determinant, compute proxy and replay are in `RESULTS_CORE.csv`.
- **Interpretation:** The Mirror representation is a restatement of ordinary finite-group indexing and loses this small-size storage/compute comparison.
- **Hypothesis:** Larger representations might amortize generator state, but native group methods remain the required control.
