# MA-573 status

- Status: **FAIL under quality/bytes gates and for Mirror-specific attribution**
- Branch: `research/ma-573-quarot-mirror-sweep-20261102`
- Protocol freeze: `f96eb43c`; A1 adds timing instrumentation only.
- Dev-selected angle: `-0.7679448709` rad from 181-point grid; fresh angle frozen.

## H / T / D / C / U

**H:** A one-angle function-preserving hidden gauge selected on development data can reduce int4 quantization distortion over identity and fixed Hadamard gauges at acceptable actual payload.

**T:** Two 8x8 linear layers; hidden Givens gauge applied as `W1R, R^T W2`; signed symmetric int4, nibble-packed, per-matrix scales; dev seeds 57301–57302 select one global angle, fresh seeds 57311–57313 run it once. Controls: identity, normalized Hadamard, same-angle direct scalar code, full precision. Full precision gauge equivalence checked before quantization.

**D — FAIL.** Fresh mean NRMSE: identity `0.421`, Hadamard `0.272`, Mirror `0.365`, direct angle `0.365`, full precision `0`. Mirror improves identity but is worse than fixed Hadamard and misses the `<=50%` full-precision payload gate: `997 B` vs `1466 B` (`68.0%`). Direct-angle control exactly matches Mirror.

**C:** A single global angle may not fit the anisotropy across both layers; fixed Hadamard is stronger on these worlds. Learned rotation search is established prior art (SpinQuant), and the direct scalar control is the same coordinate.

**U:** LLM/NLL results, GPU quantized kernels, calibration on natural models, per-layer gauges, SmoothQuant/SpinQuant search, energy and rotation-search amortization across real weights.

## Facts

- Full-precision gauge max output difference was below `3e-16` in all fresh worlds.
- Mirror/direct fresh payload is `997 B`; mean inference time `0.0062 ms` for a 512-row CPU batch. Dev angle search used 181 grid points and took `0.054 s` across both dev worlds in A1.
- Every quantized candidate uses the same 4-bit matrix code representation; scales, method IDs and angle metadata are charged inside NPZ files.
- Fresh payload bytes/hashes and direct alias were verified; one test passed.

## Interpretation

The gauge preserves the full-precision function but the selected angle does not improve over QuaRot-style Hadamard on fresh synthetic matrices and does not meet the registered storage ratio. Quantization can select equivalent representations, but this Mirror coordinate sweep did not establish an incremental benefit or natural-model result.
