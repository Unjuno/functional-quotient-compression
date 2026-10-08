# MA-461 — HyperFormer generator outputs Mirror codes

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/GENERALIZATION/RUNTIME  
Branch: `research/ma-461-hyperformer-mirror-codes-20261008`  
Base commit: `a211269`  
Prior art: PA82 (HyperFormer)

## H — Hypothesis

On a 6-task × 4-layer adapter bank with six task×layer combinations withheld, a shared hypernetwork that emits a two-value View code and decodes it through a shared matrix basis can match a direct HyperFormer-style adapter generator with lower actual payload bytes and generation compute. A direct native low-rank generator checks whether the advantage is just low-rank hypernetwork output.

## T — Frozen setup

The synthetic 4D task has 24 possible adapter functions. Eighteen task×layer combinations train both models; six combinations are withheld, while every task identity and layer position remains represented. The teacher uses a smooth rank-two family. Both generators receive the same learned 2D task and 2D layer embedding tables, use a 16-wide hidden layer, and train for 1,800 Adam updates with batches of eight combinations. Each heldout pair is evaluated on 256 input vectors. Fresh seeds 46111–46113 remain sealed. See `PROTOCOL.json` for gates and metric definitions.

PA82 establishes shared hypernetworks conditioned on task/layer/adapter-position embeddings as a native baseline. The Mirror path compresses the generator output to two coordinates plus a shared basis; the direct native low-rank generator uses the same function and is the attribution control. A full independent table is measured only on observed pairs, not used to peek at heldout target values.

## Results

Pending frozen development runs.
