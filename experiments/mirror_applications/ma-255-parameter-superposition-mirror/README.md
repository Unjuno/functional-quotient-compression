# MA-255 — Parameter Superposition with structured Mirror contexts

Status: Amendment A1 frozen before new development. Evidence lane: storage / retrieval / interference.

## H / T / D / C / U

**H:** For correlated low-rank task vectors, fixed-context Parameter Superposition (PSP) can retrieve useful task functions. Learned rank-8 Mirror coordinates must match native PSP while using a smaller complete inference payload; any Mirror-specific result must beat the byte-identical generic low-rank coordinate control.

**T:** Amendment A1 uses 16 task vectors of dimension 64 generated from an 8D shared Gaussian basis plus fixed 0.03 residual noise. Query inputs are independently sampled. Controls: exact Rademacher PSP dot-product unbinding, shared mean without task code, independent task vectors, learned 8D Mirror codes, and generic learned rank-8 codes with identical state and compute. Dev worlds 25520–25521 × seeds 0–2; 1600 Adam updates and LR {0.001, 0.01, 0.03}. Fresh worlds 25530–25532 remain sealed until dev gates pass. Actual torch payload bytes include all task state and metadata.

**D:** Pending amended development. A0 used independent vectors and an invalid broadcasting metric; its outputs are exploratory and excluded from A1. The correction and new task distribution are documented in `PROTOCOL.json` before A1 development.

**C:** Any gain may come from generic low-rank factorization rather than Mirror. The task family is deliberately correlated and synthetic; success cannot establish general model composition.

**U:** Nonlinear tasks, learned PSP contexts, natural neural-network composition, and scale beyond this low-rank diagnostic.
