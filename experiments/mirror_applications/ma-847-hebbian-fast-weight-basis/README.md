# MA-847 — Hebbian fast-weight Mirror basis

Status: SCREENING. PA229 distinguishes slow shared weights from writable Hebbian fast state. This experiment tests whether episode-specific fast matrices can be represented by a shared basis and transient coefficients.

H: basis coefficients preserve one-shot key-value recall with lower writable bytes and update work than a full Hebbian matrix.

Mirror insertion: episode coordinate m parameterizes a fast matrix M(m)=Σ_r m_r B_r. Full Hebbian outer-product state and ordinary low-rank factorization are controls. Each episode resets M/m; reset leakage is checked.

Synthetic associative memory has D=16 key/value vectors, 8 support pairs and 64 heldout queries. Development seed 13; fresh seeds 31/53. Persistent basis, transient code, writable matrix, combined bytes, query nRMSE, write MACs, and timing are recorded. This is a mechanism screen only.

C: ordinary low-rank fast state or explicit content-addressable memory may match or outperform Mirror. U: trained Hebbian Fast Weights, sequence interference, and real inference latency.
