# MA-379 — AdapterFusion bank with source-task Mirror views

Status: **FAIL on the frozen development gate; fresh seeds remain sealed.** Dedicated branch: `research/ma-379-adapterfusion-mirror-20261009`.

## H — Hypothesis

Six per-source Givens coordinates inside a shared rank-4 adapter basis can compress the AdapterFusion source bank while preserving input-conditioned target-fusion quality, and can improve the storage/quality point over hard tying and scalar gates.

## T — What was run

PA54 AdapterFusion retains independent frozen task adapters and learns a separate fusion mechanism. This CPU screen used eight rank-4 source adapters and four input-conditioned fusion tasks over a frozen 16D feature interface. Compared: independently parameterized adapters, one tied adapter, scalar gates, unrestricted 4×4 source coefficients over shared factors, and per-source Givens views over shared factors. Source bank updates: 1,200; fusion-router updates: 400 per target; Adam, LR .01, batch 64. Two development worlds (37901, 37902) were run under the pre-development protocol commit `bfe4cca9`. Every source/code/router/metadata object was charged in actual compressed NPZ payload bytes. Fresh IDs 37911–37913 were not opened.

## D — Decision

**Fact:** Independent bank source NRMSE was <1.4e-7 in both worlds. Mirror source NRMSE was <2.4e-7 and mean target-fusion test NRMSE was .000618/.000828, effectively equal to the generic 4×4 coefficient control (.000618/.000828). Mirror payloads were 4,634/4,642B versus independent-bank 7,617/7,607B (60.8%/61.0%); the frozen limit was 60%. Scalar gates used 4,473/4,476B but target fusion NRMSE was .868/.746. Generic coefficients used 4,977/4,970B at the same quality as Mirror. All 10 stored payloads passed byte/hash and metric replay; four tests pass.

**Interpretation:** Mirror recovers the deliberately aligned adapter orbit and gives a substantial quality improvement over a scalar gate for only 3.6–3.7% more total payload. It also saves about 6.7% versus the unrestricted coefficient bank at equal measured quality. Yet it misses the preregistered 40% total-payload reduction versus independent AdapterFusion by about one percentage point, so the registered outcome is FAIL. The fixed source-bank MAC proxy was 832 for Mirror versus 1,024 independent and 704 generic coefficients; the target-router proxy was 1,024 for all methods. Average training wall time was 1.65s Mirror, 1.62s independent and .76s generic coefficients on this CPU. No robust deployment-latency claim follows from the single-batch throughput sample.

## C — Strongest counter-hypothesis

The teacher bank was intentionally generated from the same shared-basis Givens orbit under test, so it is a favorable upper-bound feasibility fixture. Real source adapters may not share that orbit. The generic coefficient control matched Mirror quality, and common fusion-router bytes diluted adapter-bank compression in the total payload.

## U — Unresolved

No pretrained Transformer, natural language task, published AdapterFusion benchmark, target-domain transfer study, larger adapter bank, or device latency test was run. Fresh worlds are sealed because the total-payload gate failed. This is neither a capacity claim nor a natural-task claim.

## Evidence labels

- **Fact:** Stored per-seed metrics and actual payloads are under `results/development/`; verification hashes and replays are in `VERIFICATION.json`.
- **Interpretation:** Within this aligned proxy, Givens views trade a small amount of bytes for a large quality recovery over scalar gating, but fail the frozen total storage target.
- **Hypothesis:** A natural AdapterFusion bank with clustered low-dimensional task updates could retain this tradeoff; test that only under a new protocol with task-identity-disjoint development/fresh adapters.
