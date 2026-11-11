# MA-844 — Fisher-metric distance for task update code assignment

Status: SCREENING. Prior art: PA227 (AI Engram) and PA178 (Fisher-weighted merging).

H: Fisher-weighted Mirror code assignment preserves heldout function updates with >=25% fewer complete bytes and lower Fisher-weighted error than Euclidean assignment, beyond native Fisher k-means.

We construct six synthetic update clusters under an anisotropic diagonal Fisher metric, fit the codebook on 24 tasks, and evaluate 48 heldout tasks. Code indices, centers, Fisher values and metadata are all serialized and charged. Fisher k-means/Mirror use 1,995 B vs 6,588 B independent (69.7% fewer), but nRMSE .04609 misses the .02 gate; Euclidean k-means has the same error and retrieval. Fisher assignment uses 2x the distance-MAC proxy. Mirror exactly aliases native Fisher k-means. This is not trained PEFT or a real Fisher estimator. See [STATUS.md](STATUS.md).
