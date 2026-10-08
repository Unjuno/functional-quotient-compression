# MA-375 — One-shot supernet with architecture-path Mirror correction

Status: FAIL; frozen development gate did not pass. Dedicated branch: `research/ma-375-oneshot-subgraph-mirror-20261008`.

## H — Hypothesis

A Givens code per one-shot DAG path will improve correlation between shared-model validation path rankings and independently trained child test rankings over both uncorrected shared weights and a byte-matched scalar path code.

## Mirror insertion

The native method is a one-shot weight-shared DAG supernet (PA62/ENAS style). It has a stem, three residual operators, and a shared classifier. Mirror adds one persistent Givens angle for each of eight architecture paths, acting on the final hidden state before classification. The direct control uses one scalar hidden gain per path.

Eight paths represent all binary masks over three residual nodes. The path count is not treated as independent capacity. Each shared method uses the same one-path-per-update and minibatch sequence. Eight independent children provide a stronger quality/ranking reference; each child receives 1,200 updates and all its weights are charged.

## T — Frozen protocol

Two synthetic 16-input, four-class teacher worlds were used (seeds 37501/37502) with 8,192 train, 2,048 validation, and 4,096 test examples. The four methods are independent children, plain shared supernet, scalar-per-path, and Givens-per-path Mirror. Each runs for 1,200 Adam updates at 1e-3. The independent bank updates all eight child models each optimizer step; each shared method samples one child path per step.

Inference states are deterministic ZIP/NPY FP16 archives. Payload bytes include weights, codes, path IDs, metadata, and archive headers. Every method/path is measured for NLL, accuracy, MAC proxy, and throughput after loading its serialized payload. Rank correlation compares the shared method's validation NLL order with the independent children's test NLL order.

## D — Decision

**Fact:** Independent path-test NLL range passed the preregistered minimum in both seeds (0.0276 and 0.0203 nat). Mirror validation-to-independent-test Spearman was -0.1905 and 0.1190, versus plain supernet 0.0476 and 0.1429, and scalar control -0.1667 and -0.2143. It missed the required +0.20 gain over both controls in both seeds. Mean validation NLL was 0.2352/0.3354 for Mirror versus independent 0.2114/0.2983, so the 0.03 nat quality margin also failed in seed 37502. Mean test NLL was 0.2463/0.3089 for Mirror, 0.2366/0.3076 for plain supernet, 0.2320/0.2900 for scalar, and 0.2227/0.2752 for independent children.

Mirror used 8,737B, equal to scalar, 2.8% more than plain supernet (8,499B), and 78.0% less than the independent bank (39,783B). All eight payloads passed size/hash/reload/metric replay; four tests passed. Fresh seeds remain sealed.

**Interpretation:** Path-specific Givens codes did not reduce measured one-shot ranking bias or improve mean child quality. Most storage reduction came from base weight sharing; Mirror added bytes and did not beat the equal-byte scalar control. Independent children received 1,200 updates each, eight times the shared model-example exposure, so their quality is a compute-heavier reference.

## C — Strongest counter-hypothesis

Independent child quality spread was only just above the preregistered threshold and there were only eight paths, making Spearman rankings noisy. The shared supernet saw each path for about one eighth of its updates; ranking bias may come from path undertraining, which a final Givens view does not address.

## U — Unresolved

No real NAS benchmark, larger path set, longer one-shot training, ranking uncertainty intervals, or hardware deployment test was run. This synthetic fixed-budget result establishes neither architecture-search utility nor capacity.

## Evidence labels

- **Fact:** Measurements, actual bytes, hashes, per-path metrics, and runtime are in `RESULTS_CORE.csv` and `results/development/`.
- **Interpretation:** Mirror did not correct the measured shared-weight ranking bias under this setup.
- **Hypothesis:** Path undertraining or low ranking spread, rather than code geometry alone, may dominate; this needs a separate registered test.
