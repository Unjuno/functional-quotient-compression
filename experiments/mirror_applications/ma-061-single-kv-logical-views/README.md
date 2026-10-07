# MA-061 — one KV head to many logical KV heads

Status: SCREENING. Branch `research/ma-061-single-kv-logical-views-20261007`.

## H

One physical K/V projection and cache stream, expanded into four logical query-head roles through small Mirror coordinates, may preserve an aligned attention teacher while reducing actual persistent cache bytes relative to MQA/GQA/MHA.

## T

Synthetic 16D attention, four query heads of dimension 4, sequence length 8. Compare full MHA, GQA-2, MQA, rank-1 per-head K/V residuals, and MQA-cache Mirror views. Teacher modes use shared K/V plus views or independent K/V projections. Record full serialized model bytes and exact persistent K/V bytes per sequence.

## Decision

FACT: development/fresh pending. INTERPRETATION: pending. HYPOTHESIS: pending. BOUNDARY: synthetic attention/cache only; no language NLL claim.
