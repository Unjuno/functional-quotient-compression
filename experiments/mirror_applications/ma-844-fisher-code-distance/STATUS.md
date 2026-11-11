# MA-844 status

Status: **FAIL** for the preregistered quality/Mirror-specific gate; scoped synthetic metric screen. Fresh remained sealed after quality miss and exact native Fisher-control alias.

## H / T / D / C / U

- **H:** Fisher-weighted code assignment would preserve heldout functions with >=25% fewer bytes, lower Fisher error than Euclidean assignment, and beat native Fisher k-means.
- **T:** One synthetic seed; 24 codebook-fit tasks and 48 heldout, D=32, six constructed clusters, diagonal Fisher spectrum .1–10; deterministic farthest-point initialization plus Lloyd updates; no optimizer updates. Focused test: 1 passed.
- **D:** FAIL. Fisher Mirror and native Fisher k-means both use 1,995 B vs 6,588 B independent (69.7% fewer) but Fisher nRMSE is .04609, above .02. Euclidean k-means has the same .04609 Fisher nRMSE and same 1.0 cluster retrieval. Fisher assignment requires 27,648 distance-MAC proxy vs Euclidean 13,824. Shared mean uses 954 B but Fisher nRMSE .8941. Fresh remained sealed.
- **C:** For these compact, well-separated synthetic clusters, both metrics select functionally equivalent code assignments. The Fisher metric doubles assignment work without lowering functional error. Codebook quantization still misses the strict quality tolerance.
- **U:** No learned Fisher estimator, trained PEFT, neural task adaptation, different Fisher spectra, fresh seed, or online deployment measurement.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror and native Fisher k-means hashes, bytes and metrics match exactly. Fisher and Euclidean k-means have identical heldout errors/cluster retrieval in this seed.
- **Interpretation:** Fisher geometry changes the distance calculation but does not improve this task bank's storage/quality frontier; it costs twice the assignment distance operations.
- **Hypothesis:** Information geometry may matter when task differences align strongly with sensitivity directions; this synthetic world provides no evidence of that benefit.
