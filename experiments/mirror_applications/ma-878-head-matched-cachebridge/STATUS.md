# MA-878 status

- Status: SCREENING
- Branch: `research/ma-878-head-matched-cachebridge-20261111`
- Base commit: `e705db936`
- Protocol freeze: `e6a164427071f3b424d823e727e4a1f96eb6e9c3`
- Development run: complete; rank 32 selected by the frozen minimum-NLL rule.
- Development artifact SHA256: `f62469aaecfd09dba436b54ef29cc8d8d4d367bf7ba07e3f02caf48476208e40`
- Fresh settings freeze: pending
- Fresh/audit opened: no
- Results committed: pending
- Verification committed: pending

## Current scope

Pythia-70M→SmolLM2-135M; fixed [0,4,7] KV-head support versus attention-weighted injective support selection. The same position-free RoPE-stripped map interface is used for both.

## Blockers

None identified. CPU-only; CUDA is unavailable.
