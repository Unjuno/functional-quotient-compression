# MA-560 status

- Status: **FAIL under the frozen total-payload gate; PROMISING quality result on the aligned synthetic width orbit**
- Branch: `research/ma-560-universally-slimmable-mirror-20261102`
- Protocol freeze: `1831195c`; A1 timing amendment fixed before identical replay: `f7da2286`
- Worlds: development 56001–56002; fresh 56011–56013
- Tests: 4 passed; 160 held-out metric rows and all payload byte/hash records replay exactly

## H / T / D / C / U

**H:** A continuous width-dependent Givens coordinate on one shared width-elastic network can preserve functions at unseen widths and improve quality/storage over prefix-only US-Net and a generic direct coefficient width code.

**T:** Fixed 16→32→16 linear network; train widths 8/16/24/32; held-out widths 4/12/20/28; 512 training and 512 held-out Gaussian inputs per width. Two development worlds selected no hyperparameters; three fresh worlds ran the frozen setting once. No optimizer updates. A1 replay changed timing instrumentation only; A0 artifacts remain available.

**D — FAIL overall, with a scoped quality signal.** Mirror's fresh held-out NRMSE mean/max was `7.51e-9 / 3.19e-8`, versus prefix US-Net `0.2876 / 0.5404` and direct affine sine/cosine coefficients `0.01123 / 0.03466`. Independent full width maps scored `2.48e-8 / 2.84e-8`. Mirror total payload was 5,396 B versus 9,032 B independent (59.8%, a 40.2% reduction); the frozen gate required <=50%, so the storage gate failed. The byte-near direct control used 5,410 B, only 14 B more, but had higher error. The frozen Mirror-specific byte gate is not established at matched quality.

**C:** The teacher was deliberately generated from an affine Givens angle law, exactly the structure supplied to Mirror. This is a favorable orbit-aligned mechanism test, not evidence that arbitrary width-conditioned tasks admit this code. The direct coefficient control was linear in width and approximated the trigonometric orbit; a richer generic generator could close the quality gap.

**U:** Trained US-Net or Transformer weights, natural tasks, learned code acquisition, near-convergence capacity, widths outside the tested grid, and robust accelerator latency/energy.

## Facts

- Fresh results cover 12 world×width combinations per method. Mirror stays below `3.2e-8` NRMSE on every held-out width; the no-view prefix baseline reaches up to `0.5404`.
- Actual serialized payloads including shared maps and metadata: prefix 5,138 B; Mirror 5,396 B; direct coefficients 5,410 B; independent full maps 9,032 B.
- Fresh mean MAC proxy/example (excluding one-time map reconstruction): prefix 1,024; Mirror 1,040; direct coefficient 2,048; independent 512. These proxies do not include the full code decode.
- A1 separates per-width map reconstruction from 20 repeated batch forwards. Fresh means were 0.139 ms decode + 0.00769 ms/512-example batch for Mirror, 0.145 ms + 0.00776 ms for direct coefficients, and 0.147 ms + 0.00781 ms for prefix US-Net. Independent full maps were stored precomputed, so their decode time is not comparable to reconstructing a shared map.
- A0/A1 outputs, payload sizes and SHA-256 hashes are identical. Verification replays all 160 rows exactly; four tests pass.

## Interpretation

The two-scalar angle law transfers across fresh widths in this intentionally matched synthetic family. It reduces total storage relative to one full matrix per width, but misses the preregistered 50% frontier and offers only 14 B total savings over the generic coefficient payload. The measured quality gain is therefore a scoped fit to a known rotation orbit; it does not establish broad width multiplicity or a general US-Net improvement.
