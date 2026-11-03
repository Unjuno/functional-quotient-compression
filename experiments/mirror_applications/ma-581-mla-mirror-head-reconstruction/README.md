# MA-581 — MLA latent with Mirror head reconstruction codes

Status: FAIL
Prior art: PA115 MLA

## H

A shared rank-128 per-layer latent may reconstruct most Pythia K/V state; a rank-4 code per head/KV role over one shared residual dictionary may restore quality with lower total cache cost than independent per-head residual bases.

## T

Fit centered joint K/V PCA on four 64-token WikiText train prefixes, retain 128 dimensions, then fit shared and per-role rank-4 residual bases. Evaluate next-token NLL on 16 valid-text queries after reconstructed cached prefixes. Charge PCA projection state, residual basis state, and per-session latent/code bytes separately; report deployment bytes at 1, 4 and 8 active caches. Compare FP16 cache, MLA rank128, shared residual Mirror codes, the exact native shared dictionary, and private per-head residual bases. Fresh test split remains sealed behind the development gates.

This screen applies PCA after Pythia has generated K/V. It does not replace the model's K/V projections or establish an efficient MLA serving kernel.
