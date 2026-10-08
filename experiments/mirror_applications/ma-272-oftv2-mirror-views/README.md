# MA-272 — OFTv2 input-side Mirror transform

Status: **FAIL for Mirror-specific/runtime Pareto gain (ordinary scalar generation matches exactly and input-side latency gate missed)**. Evidence lane: MECHANISM / STORAGE / RUNTIME. Base: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: A shared skew generator with per-task addresses can apply orthogonal task views input-side with fewer bytes than independent OFTv2 transforms, while finite Cayley-Neumann truncation may trade transform time for bounded function error. A scalar-generated shared skew matrix is also a simple rank-one control and must use identical serialization.

> **Mirror insertion:** this experiment adds a task scalar `m_t` to one shared skew-symmetric matrix S, applying Q(m_t S) to activations before the shared weight matrix.

PA22 motivates input-centric orthogonal adaptation and Cayley-Neumann. Compare independent OFTv2 skew matrices, shared Mirror addresses, ordinary scalar-times-shared-matrix control, full Cayley, truncated Neumann, and weight-materialized reference.

## T

Synthetic 16D regression block with six task transforms. Shared skew generator S is scaled to spectral norm <=0.2; task transforms use Cayley(m_t S), |m_t|<=1. Compare exact solve, Neumann inverse truncations K=2/4/8, input-side xQ then W, and materialized QW. Development seeds 33/47; initial fresh 133/241/337/449 (invalid runtime accounting, preserved); first corrected audit 149/251/353/457 (native OFTv2 state still redundantly factored); final audit 157/263/359/461 locked. Measure output MSE, Q orthogonality, cycle error, serialized bytes, transform application time and active MACs. Zero optimizer updates.

## Gates

PASS: shared View within 1e-5 relative output MSE of exact input-side OFTv2, <=80% bytes, and at least 10% lower measured transform time than materialized weights on >=3/4 seeds. FAIL: no byte/runtime Pareto improvement vs the scalar-generated simple control or truncation error >1e-3 at K=8.

## Accounting

Charge W, independent skew parameters or shared S + task scalars, all reconstruction metadata. Actual NPZ bytes authoritative. Time exact input-side vs materialized dense map and report Neumann order separately; CPU small-matrix timings are exploratory.

## Results — H / T / D / C / U

FACT: Final corrected audit seeds 157, 263, 359, 461: aligned exact Mirror Cayley output relative RMSE was zero with 3,786 payload bytes; independent OFTv2 stored each actual task skew at 8,302 bytes, also zero error. The ordinary scalar-times-shared-skew control was mathematically identical and also 3,786 bytes. Materialized QW was numerical-zero error at 12,546 bytes. In independent-skew stress, shared Mirror and simple control both had relative RMSE 0.1478. Neumann K=2 aligned relative RMSE was 7.47e-4; K=4 1.37e-5; K=8 5.0e-9. K=8 orthogonality error was near 2e-15. Mean CPU transform times: exact shared Mirror 0.335 ms, materialized 0.138 ms, Neumann K=2 1.309 ms, K=4 1.164 ms, K=8 2.290 ms. The first two fresh result sets are preserved as invalidated accounting versions; final corrected audit used new seeds. Roundtrip/skew reconstruction test passed.

INTERPRETATION: A shared skew generator compresses an aligned task orbit, but ordinary scalar generation gives the same functions and bytes. Input-side transforms did not improve latency on this CPU screen; Neumann iteration adds cost. Arbitrary task skew matrices retain a quality gap.

HYPOTHESIS: Shared generators can save bytes when task transforms truly lie on one low-dimensional orbit; address codes alone do not create unrelated transforms.

COUNTER-HYPOTHESIS: Small CPU matrices and Python loops may mispredict optimized GPU kernels.

UNCONFIRMED: GPU latency, production transformer blocks, nonlinear downstream quality, learned fine-tuning and deployment throughput.

Decision: FAIL for Mirror-specific/runtime Pareto gain. The scalar-generator control matches exactly; input-side latency gate missed.

## Protocol variant reconciliation

The distinct `protocol_variants/trained_input_crossover/` directory retains a trained input-centric screen on three fresh worlds. It achieved 3,272B and near-zero MSE, versus 7,175B dense OFTv2 and 6,033B independent, but eager CPU throughput was 0.77–0.83M examples/s, below OFTv2 at 1.16–1.24M. This was a bounded storage/quality result and used a different task/protocol.

The root corrected post-fit audit is decisive for the registered Mirror-specific/runtime claim: ordinary scalar-times-shared-skew generation has the same function and 3,786B payload as Mirror, and exact input-side Mirror averaged 0.335ms versus 0.138ms for materialized weights. The audit preserves two invalidated accounting versions and reports only the final corrected seeds 157/263/359/461. Therefore the aligned compression observation is retained, but no Mirror-specific gain or input-side runtime win is established.

Source branches: corrected audit `research/ma-272-oftv2-mirror-views-20261008` (verified result commit `1e949b6b0981900c4e41d3b36efbebd06998e1ff`); trained input-side cross-over `research/ma-272-oftv2-mirror-input-views-20261008` (report commit `501d9eb18a580d378178395204c87f8b5fb39665`).
