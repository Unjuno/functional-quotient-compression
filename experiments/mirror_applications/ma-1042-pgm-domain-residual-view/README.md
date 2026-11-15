# MA-1042 — PGM-index domain residual View

Status: SCREENING  
Evidence lane: INDEX / CORRECTNESS / STORAGE / COMPUTE / RUNTIME  
Prior art: PA325; official PGM-index C++ source pinned in `PROTOCOL.json`.

## Hypothesis

A small task code can turn one canonical ordered key set and one dynamic PGM-index into four exact logical predecessor maps when each domain is a monotone affine image of the canonical keys. The comparison charges all explicit key/value arrays, domain codes, and reconstruction metadata.

> **Mirror insertion:** this experiment adds per-domain affine code `m` to the query and result interface of one shared DynamicPGMIndex so four ordered domains can use one physical ordered key set and predictor.

## Prior-art delta and controls

The official PGM-index already provides approximate piecewise-linear predictions, bounded correction ranges, predecessor lookup, and insert/delete updates. This experiment adds only a domain coordinate before the native query path. Controls are four independent official PGM-indexes, a shared hard-tied canonical index, a matched ordinary native affine query transform, and exact ordered-map predecessor outputs. Any match by the ordinary affine control is a failure of Mirror-specific attribution.

## Freeze

Two development seeds select epsilon from the frozen grid under an exactness constraint; AM-01 records that the official dynamic API does not expose correction-window width. Three disjoint fresh seeds are reserved for audit. Each world inserts and deletes 128 keys before queries. All serialized keys, task coordinates and reconstruction metadata are charged; PGM segments are deterministically rebuilt and their RAM bytes/build cost reported.
