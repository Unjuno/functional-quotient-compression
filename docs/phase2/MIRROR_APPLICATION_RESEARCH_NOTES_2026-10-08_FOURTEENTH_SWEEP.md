# Mirror Application Research — Fourteenth Literature Sweep (2026-10-08 JST)

Type: primary-source research, native-baseline map, and experiment design. **No new Mirror model training, fresh-world tests, benchmark speedups, or adoption claims** are produced in this sweep.

## Verified registry additions and scope

Previous baseline: MA001..1045, PA01..PA325, 47 report-backed experimental outcomes (29 PROMISING / 18 FAIL).
New primary prior art: **PA326..PA350 (25)**.
New experiment candidates: **MA1046..MA1095 (50)**, all UNTESTED, comprising 40 P0 and 10 P1.
New totals: **1095 MA** = 581 P0, 411 P1, 103 P2; 1048 UNTESTED / 29 PROMISING / 18 FAIL.
The canonical next experiment remains **MA-255**. This research intake does not interrupt or reassign existing worker claims.

Central Mirror hypothesis:
`F(x; theta) -> F(x; theta, m)`,
where `m` is the low-description functional parameter whose marginal usefulness must be established beyond the native conditioning/adaptation method. A sensor ID, historical series ID, table-free hash ID, wavelength, or existing LoRA coefficient is **not** a new Mirror mechanism merely because it is small. Direct priors are treated as strong controls.

## CC. Universal time-series forecasting already shares physical backbones

Main sources:
- **PA326 Chronos**: scales and quantizes real-valued time series as tokens for pretrained probabilistic forecasting;
- **PA327 TimesFM**: zero-shot forecasting across history lengths, forecast horizons and sampling granularities;
- **PA328 Moirai**: universal multivariate/frequency forecasting with trained handling of heterogeneous variates and distributions;
- **PA329 PatchTST**: channel-independent patch encoder already shares transformer weights across variates;
- **PA330 TimeMixer**: efficient multi-scale trend/seasonal decomposition and mixing;
- **PA331 iTransformer**: variable tokens give an existing route to shared multivariate representation;
- **PA332 TRACE**: forecast-specific PEFT/selected LoRA modules and adapted heads;
- **PA333 MOMENT**, **PA334 Lag-Llama**: general-purpose temporal and probabilistic representations;
- **PA335 N-BEATS**, **PA336 DLinear**: strong low-cost basis and simple linear controls.

**New tests:** MA1046..MA1061.

The candidate can insert `m=(m_series,m_frequency,m_horizon,m_variate,m_regime)` into a shared prediction head, recurrent/attention features or a low-rank output operator. Each axis must causally change useful forecast behavior, not merely carry a known scaling normalization constant.

Most decisive cheap candidates:
- **MA1049:** PatchTST with a small variate View. Preserve its native shared patch encoder. Compare hard shared PatchTST, per-channel embeddings, FiLM, rank-1 output residual, a small independent per-channel head, and structured Mirror `m`. Test naturally heterogeneous time series, not a teacher built from the candidate rotation.
- **MA1052:** TRACE module selection with a Mirror task/horizon code on *the very same chosen modules*. TRACE native LoRA + rebuilt forecast head is the required direct control. If `m` does not reduce total bytes/compute or improve error it has no specific advantage.
- **MA1057:** train one factorized `m_{frequency} \times m_{horizon}` and hold out complete frequency-horizon combinations. Compare independent task heads and TimesFM/Moirai ordinary native handling. If all performance is due to the pretrained model, reject extra `m`.
- **MA1059:** aligned/misaligned/natural forecast-family decomposition. Include mixture ratio and private residual capacity with an actual efficiency frontier, not just a binary aligned success.

**Measurement and falsification:** Point measures MASE/sMAPE/MSE, probabilistic WQL/CRPS and interval coverage, separate training/dev/future chronological test splits, unknown series and unseen frequencies, active inference work, actual serialized inference and optional optimizer state, scaling/normalization bytes and throughput. Compare DLinear/seasonal-naive at matched history and horizon. Avoid chronological leakage and evaluation from overlapping windows. A longer future horizon has different intrinsic uncertainty; never compare its raw MSE to a shorter horizon as a direct model win.

## CD. Recommendation embeddings: previous compression is already extremely strong

Main sources:
- **PA337 DLRM**: high-cardinality ID embedding tables are a substantial physical storage/memory traffic source.
- **PA338 DHE**: deterministic multi-hash ID vector plus neural decoder eliminates a learned per-ID table. This is a **direct antecedent** to one decoder serving many item embeddings.
- **PA339 QR compositional embeddings**: combinations of small complementary partitions already give a compact code representation for IDs.
- **PA340 TT-Rec**: tensor-train embedding tables with optimized TT-EmbeddingBag kernels and caching; strong compression and practical lookup frontiers.
- **PA341 VQ-Rec**: discretized item semantic codes and cross-domain transfer.
- **PA342 HSTU**: sequential large-scale recommender with context/user-history functions.
- **PA343 MMoE**: shared physical task experts with per-task gates.

**New tests:** MA1062..MA1078.

Here Mirror m could encode `(field,task,domain,hot_id,session)` over a common embedding generator, tensorized table or expert-bank operator.

**Crucial distinction:** DHE's hash inputs are a **deterministic function of ID** and require no per-ID learnable state. A Mirror method storing a separate `m_i` per item may be *less compressed* after accounting for item count and hash generation. TT-Rec may be faster after kernel fusion/caching even with more raw arithmetic. A fair result must beat DHE/QR/TT-Rec, not only a dense uncompressed DLRM.

Most decisive candidates:
- **MA1063:** DHE plus structured field/domain Mirror code versus unchanged DHE, native ID hash, field-ID embedding, scalar gate/FiLM and rank-1 adapter. Preserve deterministic hashing, train/serve consistency and total DHE decoder bytes. A code-only aligned test is not a natural recommender compression claim.
- **MA1066:** insert View m into *native TT-Rec tensor cores*; report full TT core state, task codes, TTL of cached materialized vectors and actual TTEmbeddingBag GPU lookup time.
- **MA1070:** MMoE logical specialist View over native expert pool and gates. The same routing and strong native task gates must be retained for the MMoE control. Test jointly competing objectives and true task conflicts.
- **MA1072:** Zipf-frequency-specific shared/private allocator. Frequent IDs may justify private memory; rare IDs may benefit from a shared decoder. This is an empirical frontier requiring held-out cold IDs and time-based temporal drift splits, not a theorem about frequency.
- **MA1078:** common end-to-end Criteo-like table-vs-generator-vs-TT quality, total memory and P95/P99 serving performance protocol. Use open/public licensed datasets; real production identities are not needed.

**Measurement:** CTR binary logloss/AUC, sequential Recall@K/nDCG, cold and warm IDs, tensor/table/hash/state bytes, all cache and optimizer state, QPS, GPU HBM/cache misses and P95/P99 latency. Prevent leakage from item popularity/future clicks. Count cross-shard communication and host-device transfers when distributed. Distinguish byte-efficient logical embedding generation from useful independent per-ID information.

## CE. Earth-observation sensors are a direct functional-coordinate test, but DOFA is exceptionally close

Main sources:
- **PA344 SatMAE**: spectral band grouping and temporal embeddings;
- **PA345 DOFA**: wavelength-conditioned dynamic hypernetwork generating sensor-specific filters for one backbone; extremely direct prior for sensor functional parameters;
- **PA346 CROMA**: separate SAR and multispectral optical encoders with spatial-temporal alignment, contrastive objectives and fusion;
- **PA347 AnySat**: one model across many sensor modalities, ground sampling distances and resolutions;
- **PA348 Prithvi-EO-2.0**: pretrained multispectral/multitemporal foundation model;
- **PA349 TerraMind**: joint pixel/token generative representations enabling multiple modality output directions;
- **PA350 AlphaEarth Foundations**: embedding-field modeling with spatial, temporal and measurement-context data.

**New tests:** MA1079..MA1095.

Physical condition `lambda` (wavelength), sensor response, spatial sampling scale, time/season and location may each change the correct input operator. A compositional candidate could use `m=(m_sensor,m_lambda,m_GSD,m_time,m_task)`, plus sparse private sensor-residual operators when hardware-specific responses cannot be represented from a shared chart.

Prior/novelty warning: simply mapping wavelength to a filter is *exactly* the type of function DOFA already implements. A plain `m_sensor` on the same EO features is also weaker than comparing to AnySat. Mirror must either (1) reduce the physical dynamic filter/generator state at matched quality, (2) give better held-out wavelength/sensor combinations at equal memory and latency, or (3) compress many task-specific downstream heads while preserving shared input physics.

Most decisive candidates:
- **MA1079:** preserve one DOFA backbone and its training data/sensor metadata, train `m` to parameterize a shared structured filter basis in lieu of selected generated filters. Compare native DOFA hypernetwork, simple wavelength embedding, low-rank generated filter, independent sensor adapter and sparse private residual.
- **MA1080:** factor source sensor and wavelength, then hold out entire sensors/measurement-band response combinations. Wavelength alone cannot identify a SAR measurement operator.
- **MA1083:** CROMA optical/SAR transport; align exact geographic/time acquisitions and measure separate optical/SAR information. A shared output match alone does not imply missing modality information is recoverable. Include uncertainty and target signal bounds.
- **MA1090:** continuous physically calibrated wavelength spectral View versus DOFA and interpolation baselines, with unknown band center and response curves.
- **MA1095:** benchmark DOFA/AnySat/CROMA/Prithvi/TerraMind and a generic FiLM/LoRA baseline on genuinely held-out sensors/regions/years, at quality, bytes and GPU runtime.

**Measurements and controls:** land-cover/change/segmentation F1/mIoU and rare-class performance, full checkpoint and per-sensor code bytes, processor-band calibration metadata, generated kernel storage/compute, actual image patch throughput, unknown wavelength and GSD, missing bands, clouds, SAR speckle and spatial georegistration. Hold out entire geographies and periods, not adjacent tiles from the same site. Do not treat hallucinated missing sensor data as physically measured signal. Respect satellite data licenses and benchmark split boundaries.

## CF. Three quick falsification protocols suitable for a future worker

### Protocol TS-1: MA1049 patch-channel View

- Native backbone: same PatchTST weights and train/validation/future-split series.
- Controls: PatchTST ordinary channel independence, learned channel embedding, rank-1/FiLM, byte-near shared low-rank, independent channel heads.
- Candidate: one m per channel applied at the same feature interface, with exact serialized bytes.
- Challenge: unequal channels, withheld channel/task combos, temporal distribution shift and channel independence.
- Rejection: native simple code equals or beats m at similar quality-cost or m only increases capacity from extra parameters.

### Protocol REC-1: MA1063 DHE+Mirror

- Native backbone: DHE hash-ID encoding and one shared embedding decoder.
- Controls: untouched DHE, ordinary dense table, complementary QR, TT-Rec, DHE+FiLM or rank-1.
- Candidate: field/domain code m; never give a free per-ID table.
- Challenge: cold IDs, unseen category domains, shifted frequency and budget-constrained GPU lookup.
- Rejection: more m state or lower QPS with no real AUC/logloss gain over DHE.

### Protocol EO-1: MA1079 DOFA+Mirror

- Native backbone: DOFA pretrained multisensor encoder, physical band wavelength metadata.
- Controls: original DOFA dynamic filters and cheap wavelength embedding/low-rank filters; independent sensor adapters.
- Candidate: structured m over shared spectral projection/band filter tensor.
- Challenge: unseen sensors, missing bands, held-out wavelength response and distant geographic transfer.
- Rejection: no added gain over native DOFA or a simpler spectral gate at the same inference bytes/generation time.

## CG. Scientific status and worker integration rule

The total MA registry now contains **1095** entries with **1048 UNTESTED**, **29 PROMISING**, **18 FAIL**. This sweep added 50 new UNTESTED ideas and 25 primary prior works. It did not run or change any of the 47 recorded experimental outcomes.

**Do not preempt MA-255.** The new rows follow the already-locked worker and cross-domain research queues. The report is the new literature-intake context, not permission to treat source-paper performance as Mirror results.

See `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`, `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`, `experiments/mirror_applications/WORKER_QUEUE.md` and `experiments/mirror_applications/STATUS_BOARD.md`.

## CH. Mirror marginal-state break-even: a mandatory pre-screen

For each native method, denote:
- `B_native`: all physically serialized native inference state;
- `B_replaced`: native state actually removed by the proposed Mirror insertion;
- `B_basis_extra`: new shared View basis/decoder/generator state that did not exist in the native method;
- `K * B_m`: all per-logical-function persistent codes, including their own headers and indexes;
- `B_metadata`: algorithm, mapping, calibration, routing and manifest state newly required by Mirror.

Then the proposed serialized state is

`B_Mirror = B_native - B_replaced + B_basis_extra + K * B_m + B_metadata`.

A **storage-only Mirror advantage over the native method** requires

`B_replaced > B_basis_extra + K * B_m + B_metadata`.

This is not a theorem about task quality or runtime; it is an accounting identity under the chosen complete serialization format. It prevents a common false-positive failure mode:

- **DHE:** if the native architecture already has no table and m only adds one learned code per ID, `B_replaced` may be zero. There is no storage saving unless m removes part of the native generator or another paid state. An accuracy/latency benefit is still possible but must be demonstrated.
- **TT-Rec:** do not compare Mirror against the original dense embedding size once native TT cores and cache are the actual serving baseline. Count the portions of TT state truly eliminated.
- **DOFA:** if the native wavelength hypernetwork continues running unchanged, a separate spectral code/generator increases state and compute unless it replaces existing filters or measurably improves quality. Report both newly stored basis and any generated-filter latency.
- **TimesFM/Moirai/PatchTST:** if the native method already uses a single shared backbone and no per-domain head, adding K task codes cannot be called weight-copy replacement. It might be complementary functional freedom, which must be evaluated as such.

There is also a **compute/latency break-even**: the removed native lookup/generation/copy/forecast work must outweigh View decoding, code generation, extra kernel launches, memory traffic and any materialization. Report empirical latency and source-architecture-specific throughput, never infer it from FLOP proxies alone.

**Causal non-gauge check:** hold `x` and shared `theta` fixed and intervene on `m`. Show that changed `m` produces distinct useful target functions, and check whether the effect is already available through known input relabeling, time normalization, group symmetry, metadata passthrough or a cheap native gate. Combinatorial numbers of possible addresses do not prove independently stored capacity.

A worker who cannot identify `B_replaced` or the intended complementary accuracy/compute utility should not claim Mirror-specific storage value. Negative outcomes remain first-class evidence.

