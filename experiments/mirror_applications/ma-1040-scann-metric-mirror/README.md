# MA-1040 — ScaNN anisotropic metric Mirror task code

Status: SCREENING  
Evidence lane: ANN / QUALITY / STORAGE / COMPUTE  
Base commit: see `PROTOCOL.json`  
Prior art: PA318 (ScaNN anisotropic vector quantization)

## Hypothesis

Two development-selected query-length-role Givens codes can improve fresh ArgUana retrieval quality by at least 0.5 pp on one fixed ScaNN AVQ index, without duplicating the index. Mirror-specific value requires a further gain over an exactly matched native two-coordinate query transform.

> **Mirror insertion:** this experiment adds role code `m` to the query-vector input of one shared ScaNN searcher so that short- and long-query retrieval metrics can differ without rebuilding or duplicating the corpus index.

The physical shared object is a single Nomic Embed v1.5 embedding model, corpus-vector table and ScaNN AVQ index. The logical multiplicity is two retrieval functions, selected by query length. A one-index/no-view arm, exact dense cosine control and ordinary linear query-coordinate control are included. The matched linear control is expected to be the strongest falsifier because a Givens transform is a two-coordinate linear readout.

## Frozen protocol

The official BEIR ArgUana archive is pinned by SHA256 in `PROTOCOL.json`. Amendment AM-01 caps model input at 256 tokens and makes vector encoding resumable, recorded before any development metric or audit access. Query IDs are assigned to development/audit by a salted SHA256 rule independent of qrels; query length alone defines the two role addresses. The full corpus is indexed. Development chooses angles on the registered grid. Fresh audit qrels must stay closed until the code, index configuration and development freeze are committed.

ScaNN 1.4.2 CPU uses one dot-product AVQ tree, asymmetric hashing, and a fixed reorder stage. The same serialized index is used for all arms. Payload accounting charges model, codebook, index, maps, coordinates and manifests as specified in the protocol.

## Expected falsifier

A two-coordinate Givens view is expressible by a matched native query-side linear transform. Exact functional alias or matched audit ranking fails the Mirror-specific hypothesis even if query-role metric tuning helps over the no-view ScaNN baseline.

## Status

The isolated ScaNN environment is installed. The registered model and dataset are available locally / being staged. No audit metrics have been opened or inspected.
