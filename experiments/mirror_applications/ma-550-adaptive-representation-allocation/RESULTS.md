# MA-550 checked results

## Protocol execution

The source and protocol were frozen before development. Amendment 1 added F16 support to the checkpoint reader after inspecting the pinned checkpoint format but before generating any world. Development seeds 55001 and 55002 passed the frozen adaptive gates; the decision was pushed before fresh worlds 55011–55013 were opened. No tuning was performed.

## Results summary

| Method | Actual NPZ bytes | Fresh mean held-out relative RMSE | Fresh max per-function held-out RMSE | Fresh decode FLOPs proxy |
|---|---:|---:|---:|---:|
| Dense output delta | 3,220,842 | 0.002181 | 0.003976 | 804,864 additions |
| Activation code through shared W | 34,496 | 0.497123 | 0.998298 | 824,180,736 |
| One-index sparse bias | 1,760 | 0.499969 | 0.999981 | 16 scalar updates |
| Adaptive native mix | 19,682 | 0.0000288 | 0.000105 | 412,090,376 |

Across every development and fresh world, the adaptive selector chose eight activation codes for the `Wm` functions and eight sparse output biases for the one-token functions. This is ordinary sparse weight-space editing combined with a ReFT-style activation code. Its selection and output are exactly the strongest native control; there is no Mirror-specific effect.

Fresh adaptive payload was 0.611% of the fixed dense-delta payload (163.6× smaller) and 57.1% of the fixed activation-code payload. The latter has an unusable 0.497 held-out error on the sparse half. All 5 fresh/dev worlds pass the adaptive ≤1% support and held-out error gates. With the 168,144,624 B common checkpoint included, adaptive full-system payload is 168,164,306 B versus 171,365,466 B for dense output deltas, a 1.87% reduction.

The adaptive decode uses eight dense `Wm` projections and eight sparse scalar updates. The FLOP proxy is 412,090,376 for the function bank, about half the fixed all-activation code path and much higher than adding the fixed dense logit deltas. This is a bytes/compute tradeoff, not a universal Pareto improvement.

## H / T / D / C / U

**H:** Different output functions may have different cheapest sufficient representation families.
**T:** Five seeded synthetic worlds on the pinned Pythia-70M output projection; two development, three fresh; actual serialized NPZs; no training.
**D:** FAIL for Mirror-specific attribution; mixed allocation mechanism passes its synthetic gates.
**C:** The selector is conventional sparse-bias plus ReFT/shared-basis coding.
**U:** Natural task efficacy, out-of-distribution input routing, end-to-end serving latency/energy, and other models/layers remain untested.

## Evidence classes

**FACT:** Byte sizes, reconstruction errors, seeds, model hash, timings, FLOP proxies and byte-identical replay are in this directory's JSON/NPZ/CSV files.
**INTERPRETATION:** Mixed allocation is useful for the constructed linear family but equivalent to native controls.
**HYPOTHESIS:** A future natural task may benefit from adaptive representation choice if it beats direct sparse edits and standard ReFT at matched quality/compute/bytes.
