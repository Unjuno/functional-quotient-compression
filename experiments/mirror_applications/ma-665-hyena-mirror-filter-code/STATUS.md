# MA-665 status — scoped FAIL

- Decision: **FAIL for a Mirror-specific advantage.** Shared filter coordinates can compress an aligned synthetic filter family, but native PCA exactly aliases Mirror and a deterministic sinusoidal code is both smaller and equally accurate on the aligned teacher.
- Protocol frozen before development: commit `405e36fb`; hash is in `freeze.json`.
- Development seeds: 66501/66502. Fresh seeds 66511–66513 remain sealed after exact native PCA alias and native DCT dominance.
- Conditions: 16/64 logical banks × 4 channels × 128 positions, 8 sequences per filter, ranks 4/8/16, residual RMS 0/.05/.2.
- Quality: causal convolution output NRMSE on held-out Gaussian input sequences. Storage: actual serialized payload bytes. Compute: filter-generation and FFT MAC proxies, plus measured CPU inference and serialization wall times.

## H / T / D / C / U

- **H — Hypothesis:** small per-filter Mirror codes over a shared implicit filter generator can preserve useful causal filters at lower complete storage and compute than independent filters, and may beat deterministic sinusoidal and native PCA controls.
- **T — Trial:** generated 8-frequency DCT teacher filters plus orthogonal per-filter residual; 2 development seeds; 16/64 filter banks; ranks 4/8/16; residuals 0/.05/.2. Compared independent kernels, deterministic DCT codes, Mirror PCA codes, and identical native PCA codes. Evaluated causal FFT outputs, actual payload bytes, MAC proxies and CPU wall times.
- **D — Decision:** FAIL. At rho=0, rank8 outputs are exact: for N=64 Mirror PCA uses 23,756 B versus 122,288 B independent, but deterministic DCT coding uses only 8,556 B. For N=16, Mirror PCA uses 17,993 B versus 31,060 B independent (58%, missing the 50% gate), while DCT uses 2,816 B. Mirror/native PCA arrays and functions are identical. At rho=.05, even rank16 Mirror PCA has mean NRMSE .037 for N=64, above the .02 gate. Fresh remained sealed.
- **C — Strongest counter-hypothesis:** the gain is standard shared-filter coding. The task is explicitly generated from DCT functions, and the deterministic DCT control captures that prior without paying for a learned basis. PCA improves on DCT for some noisy held-out cases but is larger and exactly aliases native PCA.
- **U — Unconfirmed:** trained Hyena filter MLPs, natural language or sequence quality, nonstationary real filters, GPU FFT latency, and learned end-to-end filter-code inference. CPU timings are noisy microbenchmarks.

## Fact / Interpretation / Hypothesis

- **Fact:** 144 development rows replay exactly; payload file sizes match all recorded bytes; Mirror/native PCA arrays are exact. DCT ranks equal to the true teacher rank achieve numerical-zero NRMSE.
- **Interpretation:** filter multiplicity can be stored cheaply when temporal kernels share a known spectral basis. On this teacher, fixed sinusoidal structure beats a learned Mirror/PCA basis in bytes.
- **Hypothesis:** Mirror could matter if task-specific filters are compact in a learned basis that materially beats fixed Fourier/DCT features and native PCA at matched runtime, on natural filter families.
