# MA-487 — LISTA inference for sparse function coordinates

Status: **FAIL** (development screen; fresh data sealed)  
Branch: `research/ma-487-lista-sparse-function-inference-20261008`  
Base commit: `7f3f6586836fe3e34df6cb45ea1430871484a941`  
Prior art: PA93 (LISTA)

## H — Falsifiable hypothesis

A learned fixed-depth LISTA predictor will infer heldout sparse function coordinates at RMSE <=0.01, <=4 active atoms and <=50% of OMP3 inference operations, while improving the quality/operation frontier over direct projection plus top-3 selection.

## T — What ran

Two frozen development worlds (48701, 48702) using the MA-486 rank-32 orthonormal dictionary and three-sparse function family. LISTA depths 1/2/4 trained 100 updates on 128 functions; evaluation used 64 heldout functions. Compared independent matrices, dense projection, direct projection plus top-3, exact OMP3, and LISTA. Complete predictor NPZ bytes, active atoms, inference operation proxies, train/encode/query wall times and heldout output RMSE were recorded. Presented function matrices are transient encoder inputs and excluded from predictor payload; no function-bank storage claim is made. Serialization/hash and output replay passed; fresh seeds 48711–48713 remain sealed.

## D — Decision

**FAIL**. LISTA met RMSE, active-atom and <=50%-of-OMP operation gates, but failed the required direct-control frontier: direct projection/top-3 is more accurate, has equal or lower inference operations, and a slightly smaller payload than all LISTA depths.

## Facts

- LISTA1: RMSE 0.00158/0.00157, 3.0 active atoms, 33,906 B and 1.585M inference ops.
- LISTA2: RMSE 0.00108/0.00108, 3.0 active atoms, 33,914 B and 1.597M ops.
- LISTA4: RMSE 0.00083/0.00084, 3.0 active atoms, 33,930 B and 1.622M ops.
- Direct top-3: RMSE about 1e-7, 33,409 B and 1.573M ops. OMP3: RMSE about 1e-7, 33,402 B and 4.866M ops. Independent matrix bank: 197,230 B, zero error.
- Replay maximum metric difference was zero. Fresh data was not opened.

## Interpretations

LISTA improves inference arithmetic over iterative OMP but learns a shrinkage schedule for a task where a single dictionary projection plus top-3 is already exact. That simple native control removes the apparent LISTA advantage. No Mirror-specific function-capacity gain is established.

## C — Strongest counter-hypothesis

The orthonormal dictionary and noiseless targets favor direct projection. Non-orthogonal overcomplete dictionaries, noisy descriptors or non-linear encoders could make learned LISTA inference worthwhile, but those conditions are outside this frozen screen.

## U — Still unknown

Natural function families, learned dictionary transfer, noisy inputs, GPU latency and actual model compression are untested. The presented function matrix is an encoder input, so these predictor bytes do not compress the input bank. MA-488 instead tests whether heterogeneous functions require private atoms.
