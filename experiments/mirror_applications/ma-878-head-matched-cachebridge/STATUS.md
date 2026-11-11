# MA-878 status

- Status: SCREENING
- Branch: `research/ma-878-head-matched-cachebridge-20261111`
- Base commit: `e705db936`
- Protocol freeze: pending
- Fresh/audit opened: no
- Results committed: pending
- Verification committed: pending

## Current scope

Pythia-70M→SmolLM2-135M; fixed [0,4,7] KV-head support versus attention-weighted injective support selection. The same position-free RoPE-stripped map interface is used for both.

## Blockers

None identified. CPU-only; CUDA is unavailable.
