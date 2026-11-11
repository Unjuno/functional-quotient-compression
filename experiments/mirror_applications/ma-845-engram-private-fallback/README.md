# MA-845 — Sequential shared engram codes with private edit fallback

Status: SCREENING. Prior art: PA227 (AI Engram), PA138 (SERAC), PA139 (GRACE).

H: Store a shared code when its fixed rank-4 projection residual is <=.02; otherwise allocate a private full-vector edit. Measure retention, fallback rate and complete payload against independent/SERAC/GRACE memory.

Twenty-four synthetic memories arrive sequentially: 16 in-span, then 8 exceptions. The threshold and basis are frozen. Every code, fallback vector, key and metadata byte is charged. Synthetic only, no natural continual-learning claim.
