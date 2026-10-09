# MA-475 — SERAC external memory values as shared-basis Mirror codes

Status: SCREENING  
Branch: `research/ma-475-serac-memory-values-20261008`  
Base commit: `0606cc61d699b05bc870b80512afa68cb4de2155`  
Prior art: PA89 (SERAC)

## H — Hypothesis

A SERAC-style external counterfactual memory can keep its retrieval keys unchanged while replacing each 64-float value with an eight-coordinate code into one shared rank-eight basis. This could reduce complete memory bytes with retained held-out value quality. Ordinary PCA and int8 values distinguish Mirror-specific parameterization from generic compression.

## T — Frozen protocol

The synthetic bank contains 256 keyed counterfactual values. The shared basis is fit on 192 values; 64 are held out. Retrieval keys, nearest-key routing, query noise and radius stay fixed for every method. Full explicit values, native PCA, per-value int8 and no-edit controls are included. Fresh seeds 47511–47513 stay sealed unless all frozen gates pass. See `PROTOCOL.json` for exact byte and compute accounting.

PA89 routes matching inputs to an external counterfactual model and otherwise uses the base. MA-475 isolates storage of its counterfactual values while holding retrieval fixed; it does not model the full SERAC training or factual edit pipeline.

## Results

Pending frozen development runs.
