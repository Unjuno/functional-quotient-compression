# MA-841 — Fisher-geometric causal memory bank with Mirror basis codes

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / MEMORY CAUSALITY  
Base commit: `18e282d9`  
Prior art: PA227, AI Engram.

## Hypothesis

H: A shared Fisher-weighted basis plus compact memory codes preserves specificity, reactivation, composition and erasure necessity with >=25% fewer actual serialized bytes than independent traces and strictly beats native PCA.

## Controls and boundaries

Twelve synthetic associative memories: nine aligned in a rank-3 span and three private exceptions. We compare independent traces, shared rank3, residual ranks1/3, native Fisher-PCA factorization and a random shared basis. Fisher metric, IDs, codes and residuals are charged. This tests algebraic causal checks only; no AI Engram or cognitive claim. Rank3 is exact but costs 2,468 B vs 2,253 B independent and 6x decode MACs; lower ranks fail retrieval/composition. Mirror exactly aliases native Fisher-PCA. See [STATUS.md](STATUS.md).
