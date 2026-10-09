# MA-255 — Parameter Superposition with structured Mirror contexts

Status: Amendment A1 frozen before new development. Evidence lane: storage / retrieval / interference.

## H / T / D / C / U

**H:** For correlated low-rank task vectors, fixed-context Parameter Superposition (PSP) can retrieve useful task functions. Learned rank-8 Mirror coordinates must match native PSP while using a smaller complete inference payload; any Mirror-specific result must beat the byte-identical generic low-rank coordinate control.

**T:** Amendment A1 uses 16 task vectors of dimension 64 generated from an 8D shared Gaussian basis plus fixed 0.03 residual noise. Query inputs are independently sampled. Controls: exact Rademacher PSP dot-product unbinding, shared mean without task code, independent task vectors, learned 8D Mirror codes, and generic learned rank-8 codes with identical state and compute. Dev worlds 25520–25521 × seeds 0–2; 1600 Adam updates and LR {0.001, 0.01, 0.03}. Fresh worlds 25530–25532 remain sealed until dev gates pass. Actual torch payload bytes include all task state and metadata.

**D:** FAIL on amendment A1 development. Corrected fixed PSP unbinding has mean NRMSE 0.539/0.495 across the two dev worlds. Rank-8 Mirror gives 0.0655/0.0727 and reduces error relative to PSP, but uses 146.0%/145.8% of independent-vector bytes, missing the <=60% gate. The generic learned rank-8 control is functionally identical and byte-near-identical. Fresh worlds stayed sealed; this is neither a Mirror-specific nor a compression pass.

**C:** Any gain may come from generic low-rank factorization rather than Mirror. The task family is deliberately correlated and synthetic; success cannot establish general model composition.

**U:** Nonlinear tasks, learned PSP contexts, natural neural-network composition, and scale beyond this low-rank diagnostic. No fresh replication was opened because development failed the frozen byte gate.

## Fact / interpretation / hypothesis

- **Fact:** Corrected A1 development across worlds 25520/25521 and three seeds yielded PSP NRMSE 0.495–0.539; Mirror rank-8 0.0655–0.0727. Mirror payload was 4,729–4,755 B/task versus 3,244–3,245 B/task for independent vectors. Generic low-rank control equals Mirror predictions exactly.
- **Interpretation:** Learned low-rank task codes improve retrieval over fixed PSP on this correlated synthetic family, but their basis/code payload is larger than storing each vector independently. This is generic factorization, not Mirror-specific value.
- **Hypothesis:** A substantially lower-rank or quantized shared basis may shift the storage frontier, but that requires a new registered candidate/amendment.
