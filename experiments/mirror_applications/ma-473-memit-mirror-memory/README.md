# MA-473 — MEMIT layerwise update basis + Mirror memory codes

Status: SCREENING  
Branch: `research/ma-473-memit-mirror-memory-20261008`  
Base commit: `e92702e3efdd33b3aaf692fb94734e1dc0a28fb6`  
Prior art: PA88 (MEMIT)

## H — Hypothesis

Many fact edits across several layer-tagged weight maps may share a small update basis. Two coordinates per edit can select a combination of shared per-layer update atoms, reducing resident bank bytes compared with storing every MEMIT update tensor. Native CP and per-edit rank-two factor controls determine whether the savings are standard factorization rather than Mirror-specific.

## T — Frozen protocol

Ten keyed edits target three 12×12 layer-tagged linear maps. Eight edit signals train a shared rank-two basis/code generator and two held-out codes come from support signals. Development seeds are 47301 and 47302. Every inference payload pays all base maps and keys. Fresh seeds 47311–47313 remain sealed unless all gates pass. This is not a sequential transformer or factual editing reproduction; see `PROTOCOL.json` for gates and byte/compute accounting.

PA88 distributes calculated updates across causal MLP layers for mass factual editing. MA-473 isolates whether a fact axis can be shared across layer-tagged update tensors and compares against explicit MEMIT-style storage, native CP and per-edit low-rank storage.

## Results

Pending frozen development runs.
