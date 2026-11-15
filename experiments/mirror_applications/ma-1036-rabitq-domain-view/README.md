# MA-1036 — RaBitQ domain geometry views

Status: SCREENING. This uses FAISS `IndexRaBitQ` 1.16.0 with one bit per vector dimension to test whether a shared random-rotation basis plus a small per-domain Givens coordinate can replace independent domain rotations.

The Mirror path applies a selected angle before the shared dense basis and encodes a separate RaBitQ stream per domain. All index streams, the shared rotation matrix, angle codes and metadata are charged. A post-quantization gauge control rotates each query and decoded reconstruction together; exact Euclidean scores should be invariant, exposing the limit of isometric views.

The vectors are synthetic anisotropic clustered worlds. The experiment does not claim SIFT/BigANN performance. See `PROTOCOL.json` for gates and `ARTIFACT_PROVENANCE.json` for the FAISS version and environment.
