# MA-771 — RealNVP coupling Mirror activation views

## H — hypothesis

A shared invertible RealNVP-style activation chart plus a small task code can represent nonlinear task-specific invertible maps with fewer actual bytes than private flow weights and lower query error than linear activation steering.

## T — execution

CPU synthetic 2D tasks. The teacher was `F_m(x)=g^{-1}(g(x)+m)` with shared chart parameters α=0.7, β=0.4 and per-task 2D codes. The shared chart was fit on 8 development task IDs (500 updates, 64,000 sampled activations). Fresh seeds 77111, 77112, 77113 each used 6 new task IDs, 8 support activations/task, and 256 disjoint query activations/task. On each fresh task Mirror fit only its code (250 updates); controls fit a 2D additive shift (250), affine map (400), residual 2-16-2 MLP (600), or independent full flow (500).

Inference payloads were serialized with safetensors and include shared chart state, every task-specific parameter and serialization metadata. Exact inverse and analytic log-determinant checks were evaluated on fresh queries. The replay log and full per-task rows are in `source/`.

## D — PROMISING

All three fresh seeds passed the preregistered aligned-task gates. Mean query MSE was `1.44e-12` for Mirror, `7.51e-2` for additive shift, `5.26e-2` for affine, `2.77e-2` for MLP, and `7.23e-5` for independent flow. Mirror inverse-cycle maximum was below `7.2e-7`, and inverse log-determinant error was below `1e-6`.

Actual payload: Mirror 576 B, independent flow 1,248 B (Mirror used 46.2% of those bytes), additive shift 440 B, affine 928 B, MLP 3,792 B. The lower-byte shift missed quality by a wide margin; affine and MLP also had materially higher query error. Mean per-task adaptation time was 0.088 s for Mirror and 0.214 s for independent flow. Batched CPU inference was 0.190 µs/example for Mirror and 0.172 µs/example for independent flow, so Mirror did not improve runtime over the independent flow.

## C — strongest counter-hypothesis

The teacher is exactly aligned to the tested shared-chart-plus-translation family. The fresh query result shows that a shared chart can transfer across task IDs, but does not show that such a chart can be learned or reused for unrelated activation transformations.

## U — unconfirmed

No natural data, pretrained network, non-aligned task family, or broader chart-selection study was tested. The result does not establish capacity beyond this structured family. Inference was slightly slower than independent flow despite sharing, and the tiny eager CPU timing is not a deployment estimate.

## Fact / Interpretation / Hypothesis

- Fact: all 3 fresh seeds passed the preregistered MSE, inverse, and byte gates; Mirror used 576 B versus 1,248 B for per-task flow state.
- Interpretation: a shared invertible chart plus compact codes can recover this aligned family; this is a scoped compression/mechanism result, not evidence for arbitrary task functions.
- Hypothesis: a useful next test would learn the chart from a broader task family and include deliberately non-aligned fresh tasks, with private-flow fallback measured explicitly.
