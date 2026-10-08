# GVA Stage-0 readout-only physical cache alias: exactness PASS, Mirror-specific value FAIL

Status: **exploratory synthetic mechanism screen**, not completed MA-1162, and not updated in the canonical worker queue.

## Frozen protocol

See dedicated GitHub `PROTOCOL.json` on independent branch, committed before development/fresh run. Eight logical readout roles, 4 attention query heads, one grouped-value cache, cache prefix 128, d_value=d_content=16, d_pos=8, shared basis 3, batch 1; CPU float32, torch 2.10.0+cpu and numpy 2.3.5. Five fresh seeds 101..105, three development 11..13; no training or hyperparameter tuning. 20 warmups and 100 measured calls per fresh seed per method.

## Results

| Metric | Measured fresh five worlds | Verdict |
|---|---:|---|
| Max explicit-content-key vs absorbed-query attention score discrepancy | 1.490e-07 | algebra PASS, 5/5 |
| Max attention output discrepancy | 3.725e-08 | PASS, 5/5 |
| Physically shared prefix bytes (single canonical tensor pair) | 12,288 B | exact share valid |
| Explicitly cloned prefix bytes for eight roles | 98,304 B | 8× more physical storage |
| Max/min wrong reuse output mismatch when upstream changes | 0.04140 / 0.01983 | unsafe incompatibility detected, 5/5 |
| Native full readout bank serialized | 8,460 B NPZ | native full-M |
| Mirror shared basis plus codes serialized | 4,936 B NPZ | 41.65% smaller than full bank |
| Native factorized basis-plus-codes serialized | 4,936 B NPZ | **identical** to Mirror; M0 |
| Native precomputed-readout P95 eager CPU median | 0.066085 ms | measured |
| Mirror reconstruct-on-call P95 eager CPU median | 0.080675 ms | measured |
| Per-world P95 Mirror/native ratio median | 1.248 (range 1.126–1.306) | >1.10 threshold, FAIL |

The 8x prefix reduction compares to an *unnecessarily replicated per-role cache*, not to the strongest native GVA: **native single-cache GVA also physically aliases the exact prefix and matches that cache cost**. The native factorized code is mathematically identical to the Mirror form in this test. Consequently, while exact attention reuse and the unsafe-prefix counterexample are verified, the strong Mirror-specific Pareto gate **FAILS**.

## H / T / D / C / U

- **H:** safe final-readout m preserves exact attention and memory alias; an independent Mirror-specific benefit requires beating the native GVA shared cache *and* an ordinary low-rank factorized readout bank at comparable bytes and runtime.
- **T:** random stationary V and decoupled Kpos, 8 role-specific M, scores computed both explicit Kcontent and query absorption. Overridden past hidden tensors form the negative counterexample. The exact executed Python source is distributed in the companion downloadable ZIP linked in this conversation; its SHA256 is recorded below.
- **D:** safe numerical/copy and unsafe-detection checks PASS at 5/5; incremental Mirror storage and P95 runtime FAIL. No ADOPTED or capacity claim.
- **C:** the architectural saving comes from native GVA-style readout absorption and ordinary factorization, not an additional Mirror-specific parameter advantage in this synthetic regime.
- **U:** major errors are float32 accumulation, differing operation fusion, eager CPU timing, no true GPU decoder, no learned head weights, synthetic source distribution. Report full per-world rows and do not treat n=5 as inferential proof.

## Units and dimensions

| Symbol | 日本語の意味 | SI unit / practical | 定義・定義域・型 |
|---|---|---|---|
| B | バッチ数 | 1 | integer scalar, fixed 1 |
| H | queryヘッド数 | 1 | integer scalar, fixed 4 |
| T | キャッシュ長 | 1 | integer scalar, fixed 128 |
| d_v,d_k | Value/Key埋込次元 | 1 | integer scalars, each 16 |
| d_p | 位置Key次元 | 1 | integer scalar, 8 |
| m | 読出し機能コード | 1 | real 3-vector, one per logical role |
| M | 内容Keyの再構成行列 | 1 | real 16 × 16 matrix |
| V,Kpos | キャッシュ活性値 | 1 | real 1 × 128 × 16 / 1 × 128 × 8 tensors |
| S | 物理メモリ容量 | byte (practical non-SI) | integer scalar, measured storage |
| t | CPU実行時間 | s | nonnegative scalar, table converted to ms |
| eps | 最大絶対誤差 | 1 | nonnegative scalar difference of normalized scores/outputs |

**Dimensional check:** Q (1×4×16), M^T (16×16), V^T (16×128) multiply to score (1×4×128), equal to Q @ (V @ M)^T. Byte ratio S_single/S_copied is dimensionless; latency in seconds cannot be added to bytes. Expanded uncertainty was preregistered for accuracy metrics, but this exact algebra screen instead reports max float32 differences and all five per-world runtimes; n=5 and 100 repetitions do not establish portable latency.

## Data provenance

Source SHA256: `c45abbadf56ca83759f42e93b4f78af26ca665c9ddc874561b69eb6b6a17ef86`. Fresh CSV SHA256: `c3973191b9754741a81d52605dd1b153ce1e85a94b5097068914bfcd6521d3f2`. See `VERIFICATION.json` for all hashes, software conditions and unsupported extrapolations. The companion test source has five passing unit tests in the recorded CPU environment. These results do **not** update existing MA-1162 status because the native trained GVA experiment remains UNTESTED.
