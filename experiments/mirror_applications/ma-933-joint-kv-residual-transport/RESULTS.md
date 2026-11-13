# MA-933 results

## H — hypothesis

One source/target chart shared across residual activations, keys and values could approach independent role×layer affine quality while using ≤25% of its connector bytes, preserving held-out target next-token NLL within 0.5 nat/token of affine, and improving the relevant deployment frontier.

## T — trial

Three deterministic fresh seeds (93321, 93327, 93333), pinned TinyStories GPT-Neo 1M→8M pair, CPU. Amendment 02 trained on rows 768:1024 and evaluated on disjoint rows 512:640; 256 train and 128 validation sequences, max length 32. It fit a rank-8 shared source projection and target projection with role×layer cores/biases for residual/K/V across all 8 layers (1,200 updates per seed). The strong control fit independent affine maps for each of 24 role×layer combinations on the same train examples. Native target KV prefill served as the next-token reference. Amendment 01’s seeding defect and pilot metrics are retained in [AMENDMENTS.md](AMENDMENTS.md) and `source/amendment01_implementation_pilot.json`; they are excluded from this decision.

## D — FAIL

### Fact

| Measure | Shared chart | Independent affine | Native target
|---|---:|---:|---:|
| Mean held-out feature reconstruction MSE (residual/K/V roles and layers) | 0.3604 | 0.1831 | — |
| Residual MSE | 0.7580 | 0.4552 | — |
| Key MSE | 0.2585 | 0.0623 | — |
| Value MSE | 0.0647 | 0.0317 | — |
| Mapped-cache next-token NLL delta vs target native (nat/token) | 3.896 | 1.139 | 0.000 |
| Connector bytes (actual serialized) | 43091 | 1609494 | 0 |
| Full source+target+connector bytes | 167755967 | 169322370 | target only 115769756 |
| Mapper MAC proxy per mapped token vector | 2,880 | 16,640 | 0 |

The chart connector uses 97.3% fewer bytes than independent affine. The full deployment saves only 0.93% versus source+target+independent maps because model weights dominate. Counting the required source model, chart deployment is 44.9% larger than target-only native weights. Factorized bridge reload replay max error was 0.0. Per-seed chart fitting took 3.91s mean for 1,200 updates; the independent affine fits took 0.94s once. One-token cache evaluation wall times were 5.86s shared chart, 5.46s affine, and 5.78s native on CPU (includes source/target prefill and scoring, so not a serving throughput claim). It used 128 held-out sequences. No fused serving latency was measured.

### Interpretation

The shared chart compresses mapper state and reduces the MAC proxy, but the quality cost is too large. Mean NLL is 2.756 nat/token worse than independent affine and misses the frozen 0.5-nat gate; all three seeds fail. The shared chart also raises residual, key and value reconstruction error. Accounting for source weights removes most apparent total-byte gain against independent maps and exceeds native target-only payload. This is not a useful Pareto improvement under the tested deployment accounting.

### Hypothesis

A single shared coordinate chart may be too restrictive across activation families and depth. Separate role-private corrections or staged K/V and residual maps may be required, but would add paid state and must be tested in a new experiment.

## C — strongest counter-hypothesis

The small 1M→8M TinyStories pair, limited 256-story calibration set, one-token cache continuation, and only three seeds may underfit both mappings. Independent affine itself remains +1.14 nat/token above native. Results do not establish a general limit on joint cache/activation transport.

## U — unconfirmed

No GPU/fused cache runtime, multi-token trajectory, other model family, longer context or target-side correction was tested. Amendment 01 is implementation-pilot evidence only. No PA236 reproduction or general cache-sharing claim is made.
