# MA-585 — MLA shared latent with sparse private KV residual

Status: **FAIL for registered storage/quality frontier**  
Branch: `research/ma-585-mla-private-kv-residual-20261009`  
Base commit: `b115cb3a`  
Prior art: PA115 Multi-Head Latent Attention

## H

Shared latent Mirror reconstruction plus a small sparse private KV residual can preserve attention output quality across heterogeneous head×layer roles at fewer cache bytes than independent KV states.

## Frozen screen

Twelve head×layer roles, each with sequence-4 K/V matrices in dimension 4. Targets share a rank-2 latent structure plus controlled sparse private deviations at heterogeneity 0/.1/.3/.6. Compare shared latent only, rank-2 Mirror/direct coefficients, sparse private residual fractions 0/.05/.1/.25/.5/1, and independent K/V states. Fixed query probes score softmax-attention output error. Two development seeds, oracle SVD and top-k residual selection, zero updates. Charge all serialized K/V, bases, codes, residuals and indices; report bytes, attention output nMSE and MAC proxy.

PASS requires attention-output nMSE ≤1e-3 at ≤25% private fraction, ≥20% fewer bytes than independent cache at heterogeneity ≤.3, and Mirror-specific advantage over direct coefficients. FAIL if private state erases byte savings or direct controls alias. Fresh sealed.

## H / T / D / C / U

- **H:** Attention weighting allows shared MLA views to tolerate more KV heterogeneity than generic operators before private state is required.
- **T:** 12 roles, four heterogeneity levels, two seeds, six private fractions; 80 rows, attention-output metric.
- **D:** FAIL. At rho=0, shared rank-2 costs 2126B vs independent 2217B (4.1% saving). At rho=.1, 50% private lowers attention-output nMSE to ~.001 but costs 3762B; full residual reaches zero error at 4914B.
- **C:** Native MLA latent cache, direct coefficients and independent KV cache.
- **U:** Real attention kernels, long-context serving, GPU cache residency and language NLL.
