# MA-964 results — LIF tau/threshold factorization

## H — Hypothesis

Task×neuron membrane time constants and firing thresholds can be replaced by shared neuron bases plus small task coordinates when the teacher parameter banks are low rank. Generic banks should expose when full/private state is needed.

## T — Test

**Fact.** Fixed shared recurrent LIF core with 8 neurons, 8 input channels, 16 timesteps, seeded shared input/recurrent synapses, and 8 task identities. Three fresh worlds (96421, 96423, 96427); 128 event sequences/task/world. Compared shared per-neuron tau/thresholds, independent full tables, rank-1 SVD factors, and rank-2 SVD factors. Structured teacher tables are separable task×neuron products; generic teacher tables are independent random bounded values. SVD is an oracle compression diagnostic using the teacher tables, not learned adaptation. Every condition saw the same test event streams within a world.

Parameter state was serialized with one typed tensor stream plus JSON manifest, including synapses, task indexing, and full/factorized intrinsic parameters. The payload was reloaded and all tensors checked exact before simulated inference. Factor conditions reconstruct dense tau/threshold tables at load. No optimizer updates were run.

## Results

**Fact.** Means over three fresh worlds:

| Teacher bank | Condition | Spike Hamming agreement | Spike F1 | Payload bytes | First-spike latency error | Max pre-reset voltage |
|---|---|---:|---:|---:|---:|---:|
| Structured rank-1 | Shared parameters | 0.9698 | 0.9606 | 1,065 | 0.021 steps | 8.784 |
| Structured rank-1 | Full task×neuron | 1.0000 | 1.0000 | 1,504 | 0.000 | 8.809 |
| Structured rank-1 | Rank-1 Mirror/native | 1.0000 | 1.0000 | 1,285 | 0.000 | 8.809 |
| Structured rank-1 | Rank-2 Mirror/native | 1.0000 | 1.0000 | 1,413 | 0.000 | 8.809 |
| Generic random | Shared parameters | 0.9091 | 0.8629 | 1,063 | 0.079 steps | 10.219 |
| Generic random | Full task×neuron | 1.0000 | 1.0000 | 1,502 | 0.000 | 10.624 |
| Generic random | Rank-1 Mirror/native | 0.9154 | 0.8719 | 1,283 | 0.076 steps | 10.582 |
| Generic random | Rank-2 Mirror/native | 0.9319 | 0.8986 | 1,411 | 0.048 steps | 10.854 |

Structured rank-1 factors exactly reproduce the teacher spike traces but save only 14.6% serialized bytes versus full, below the 20% gate. The shared baseline uses 1,065 B and already reaches 96.98% Hamming agreement, exceeding the factor payload by 220 B. Generic rank-1 and rank-2 factors lose 8.5 and 6.8 percentage points of spike agreement, respectively, while saving 14.6% and 6.1% versus full. Full tables are needed for exact generic traces in this construction.

**Fact.** Mirror and ordinary native SVD factor states and spike outputs are exact aliases at both ranks. Pre-reset voltage remained finite; max over the reported runs was 10.854. Inference throughput varied substantially across identical rank-factor conditions and is not interpreted as a stable speed gain. One-time factor decode operations and runtime dense table bytes are in `RESULTS_CORE.csv`.

**Interpretation.** Aligned tau/threshold tables can be represented by low-rank task coordinates with exact event behavior, but full payload savings are modest because the shared synapse and format metadata dominate. Generic task×neuron dynamics require more than rank 2 for exact behavior. The same compression is ordinary SVD/native factorization; no Mirror-specific gain is shown.

## D — Decision

**FAIL (scoped synthetic oracle-compression screen).** Structured quality passes but actual byte savings miss the preregistered gate and the shared non-Mirror control is smaller with high event agreement. Generic rank-1/2 factors lose substantial event fidelity. No trained-network capacity, natural-event, or energy claim is established.

## C — Strongest counter-hypothesis

The 8×8 parameter bank is small, and the fixed metadata/synapse payload limits amortization. Larger task/neuron banks may provide more useful low-rank storage savings. The exact native-factor alias would still prevent attributing such a result to Mirror uniquely.

## U — Unconfirmed

No learned thresholds/timescales, optimizer convergence, STL-SNN/TACOS reproduction, natural event benchmark, or hardware energy was tested. SVD accesses the full teacher bank and therefore represents oracle compression, not an online mechanism for discovering task codes.

## Verification

Fresh command completed on seeds 96421, 96423, 96427. The common serializer was decoded before inference; every tensor round-tripped exactly. Factorized Mirror/native states and spike metrics were asserted identical. Amendment 01's post-reset voltage diagnostic is preserved but excluded; Amendment 02 reports pre-reset peak voltage.
