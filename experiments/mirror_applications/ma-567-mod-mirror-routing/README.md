# MA-567 — Mixture-of-Depths + Mirror compute/function routing

PA111 uses token-wise routing to allocate layer compute. This screen separates that compute choice from a second token attribute that chooses a function view. The CPU synthetic model uses a fixed 50% deep-token budget and compares MoD alone, a generic shared rank-2 role residual, a Mirror Givens role view, and dense full compute. No trained Transformer or language quality is claimed.
