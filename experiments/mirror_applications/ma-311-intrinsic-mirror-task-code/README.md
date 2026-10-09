# MA-311 — Mirror task code in random intrinsic subspace

Status: SCREENING. Prior art: PA33 intrinsic-dimensional fine-tuning.

## H

A task code over a small intrinsic subspace may be compressed further by representing multiple task vectors as Givens views of one shared intrinsic vector. Independent random-subspace coordinates and generic PCA test whether the effect is Mirror-specific.

## T

Synthetic regression with D=128, intrinsic d=16, eight tasks; 128 support and 256 query examples each. Compare independent least-squares intrinsic coordinates, jointly optimized shared vector + task angle (800 updates), and generic rank-2 PCA. Two strata: tasks generated from a shared Givens orbit, and independent intrinsic task vectors. Development worlds 31100–31101; fresh 31110–31112; three seeds each. The random projection U and all task codes/factors are charged in serialized payload.

## D

Pending development and fresh results.

## C

The aligned task family may be generated from exactly the same orbit Mirror optimizes; PCA or an intrinsic low-rank basis may explain any compression.

## U

Synthetic linear regression only; no pretrained language model or real task adaptation.

Development: on aligned tasks, all three approaches reach query NRMSE below 3e-7. Mirror adaptation state (shared z + eight angles) is 48 B versus 256 B independent task coordinates, but total payload including U is 4,314 B vs 4,510 B; PCA is 4,424 B. Mirror fitting takes ~0.98 s / 800 updates (~819k task-example presentations), while independent least-squares fitting takes under 1 ms. On independent tasks, direct coordinates remain near exact; Mirror NRMSE is ~0.83 and PCA ~0.63. This is a clear private-capacity boundary and compute tradeoff.

## Fresh result / decision

FAIL for practical Mirror-specific gain. Fresh aligned results reproduce near-perfect Mirror quality, but generic PCA and direct intrinsic coefficients are slightly more accurate. Since the shared projection basis dominates total storage, Mirror saves only ~4.3% overall despite compressing task-state bytes by ~81%. Mirror needs ~0.915 s / 800 updates versus <0.5 ms direct. In independent tasks, Mirror NRMSE is ~0.843 while direct intrinsic coefficients remain near exact. The independent stratum marks where private coordinates are needed.
