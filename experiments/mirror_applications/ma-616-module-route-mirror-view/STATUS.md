# MA-616 status

Stage: deterministic development screen complete; fresh remains sealed.

## H / T / D / C / U

- **H:** joint route × module View codes can represent repeated module paths more compactly than direct coefficients and independent task matrices.
- **T:** two development worlds (61601, 61602), 32 tasks each: 24 aligned Givens-view tasks and 8 unrelated tasks. Controls: route-only PathNet, one-angle Mirror, direct sine/cosine coefficients, independent matrices. No optimizer updates. Fresh seeds were not run because the frozen development byte/attribution gates failed.
- **D:** **FAIL** for Mirror-specific storage/Pareto value. Mirror and direct coefficient controls both had exact output (nMSE 0) and identical actual serialized payload (3,033 B per world); the coefficient control reproduced the exact same effective matrices. Mirror was slower in both worlds (~1.49/1.66 ms vs ~1.35/1.52 ms). Total Mirror state exceeded independent per-task matrices (3,033 B vs 2,217 B) because it paid for the module bank and private fallbacks. Route-only PathNet was 2,085 B but had aligned mean nMSE 1.48/2.23.
- **C:** the phase code is exactly the ordinary paired sine/cosine coefficient address; archive alignment removes the nominal tensor-byte difference. Module-bank and private-state overhead dominate the end-to-end payload.
- **U:** learned routers, non-oracle routes, nonlinear trained modules, larger banks, fresh replication, and optimized fused kernels.

## Fact / interpretation / hypothesis

**Fact:** in both development worlds, 8/32 unrelated tasks required exact private matrices. The one-angle Mirror and direct coefficient codes produced identical predictions and 3,033 B serialized payloads. The independent payload was 2,217 B. No training updates or fresh data were used.

**Interpretation:** routing and views can recover the deliberately aligned functions, but this representation did not save actual total bytes or compute against its nearest native control. The exact coefficient alias means the observed functional multiplicity is not Mirror-specific.

**Hypothesis:** a different module bank layout, materially more reuse per physical module, or a non-aliased native router could change the frontier; none is established here.
