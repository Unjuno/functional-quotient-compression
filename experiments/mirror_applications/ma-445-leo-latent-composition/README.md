# MA-445 — LEO latent composition of known skills

## H — Hypothesis

For an additive skill family, summing two independently learned 2D skill coordinates and decoding once will match direct task-vector arithmetic on held-out skill pairs within 1.10x query NRMSE, while using fewer actual serialized inference bytes at N=20. This is falsified if any fresh world misses either gate.

## T — Planned test

A synthetic 8D linear regression task has six skill directions. Six pairs are held out from meta-training and development. Controls are shared base, direct task-vector arithmetic, generic LEO decoder with composed codes, structured Mirror decoder with composed codes, and independent per-pair fitting as an upper reference. Development worlds 44500–44501 select the outer learning rate; fresh worlds 44510–44512 are locked before execution. Every skill code, decoder, delta, index, and metadata byte is charged.

## D — Decision

Pending. The protocol is frozen; no result exists yet.

## C — Strongest counter-hypothesis

The pairwise function is exactly represented by direct addition of skill deltas, so a latent decoder may add bytes and approximation error without improving composition.

## U — Unknowns

Whether the latent factors are identifiable, whether unseen pairs compose under a shared decoder, and whether any quality gain survives actual byte and adaptation-compute accounting.
