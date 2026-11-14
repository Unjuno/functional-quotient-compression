# MA-1009 — DUNE latent-prior scanner Mirror chart

Status: **FAIL (bounded synthetic development gate)**  
Evidence lane: **mechanism / serialized storage / CPU compute**  
Branch: `research/ma-1009-dune-representation-chart-20261114`

## H — Hypothesis

One scanner-specific Givens coordinate, conjugating only the shared DUNE-z latent regularizer, will improve scanner-B development NMSE by at least 5% over native shared DUNE-z and same-model latent FiLM, remain within 10% of target full fine-tuning, and reduce the total serialized two-scanner payload by at least 20% versus two independent scanner models. The decoder VJP and exact image-space measurement constraint stay identical across conditions.

## T — Test

PA304 describes DUNE as representation-space unrolling with the exact decoder-Jacobian data-consistency gradient `J_D(z)^H A^H(A(D(z))-y)` and shared encoder/decoder/regularizer in DUNE-z. I inspected arXiv:2606.21602 and public repository commit `8e62d1b9ca193060d09f5ac4e866dc4a6af28347`. The observed repository contains only a README; the paper says code will be available upon publication. There is no declared source license or model checkpoint. `source/dune_screen.py` independently implements the published equations; it copies no upstream code.

The CPU screen uses analytic 32×32 complex phantoms, two known smooth coil maps, and a 4x Cartesian mask with eight center lines. There are 896 complex acquired samples for 1,024 complex image values, so the inverse problem remains underdetermined. A shared 619-parameter DUNE-z model trains on 24 scanner-A examples (48 updates). Eight scanner-B support examples update either one Givens angle, eight FiLM coefficients, or all 619 model parameters for 48 updates. Sixteen separate scanner-B phantoms form the development set.

Each latent step computes the exact decoder VJP. Final images are orthogonally projected onto `A(x)=y` with QR of the explicit complex SENSE operator. All models use the same projection. Acquired-sample residual after re-encoding is below `8e-8` in every condition. QR construction takes 0.805 s and uses 42,212,352 B transient memory per distinct sensitivity-map/mask pair. Cached inference takes 7.1–9.4 ms/case. The prior uses 589,824 CNN MAC/step, or 1,769,472 for three steps, plus five forward and four adjoint FFT calls.

No fastMRI acquisition, DUNE checkpoint, clinical scanner data, or patient data were available or used.

## Development results

| Method | NMSE ↓ | PSNR dB ↑ | global SSIM ↑ | Actual one-model payload |
|---|---:|---:|---:|---:|
| Native shared DUNE-z, frozen | 0.011696415 | 33.9836 | 0.990572 | 5,965 B |
| Ordinary latent FiLM | 0.011696752 | 33.9836 | 0.990570 | 6,469 B |
| Latent Givens Mirror | 0.011696345 | 33.9837 | 0.990572 | 6,217 B |
| Scanner-B full fine-tune | 0.011695722 | 33.9847 | 0.990567 | 5,965 B |

For identical two-scanner serialization framing, complete payloads are: hard shared DUNE **5,965 B**; shared DUNE plus Mirror coordinate **6,217 B**; plus FiLM code **6,469 B**; two independent scanner models **10,419 B**. Mirror saves 40.33% versus duplicating a full model, but costs 252 B over hard sharing at effectively the same quality. Hard sharing already saves 42.75% versus two models. Mirror's small apparent edge over FiLM is 0.00348% relative NMSE; cached Mirror inference is slightly slower. All totals use the same payload keys and metadata.

## D — Decision

**FAIL at the development gate.** Mirror NMSE improves by only 0.000602% over hard sharing and 0.003480% over FiLM, far below the frozen 5% threshold. It is within 0.005320% of the scanner-B full fine-tune result, but that fine-tune is slightly worse than the native shared model on this small support set. The two-scanner Mirror package is larger than hard sharing. All methods pass the exact measurement residual gate. Fresh seeds remain sealed.

## C — Strongest counter-hypothesis

Scanner-specific sensitivity maps are already supplied to the exact forward model, so the measured operator accounts for the shift. The native shared prior is sufficient; the Givens code makes only a numerically tiny change. This synthetic task may also omit real sequence and scanner prior shifts.

## U — Unconfirmed

No acquired-data, released DUNE code/checkpoint, scanner, fastMRI, anatomy-generalization, or clinical result. One source split, one coil-map shift, and one small CPU model only. No fresh result is claimed.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror NMSE is 0.011696345, native shared is 0.011696415, and FiLM is 0.011696752. Matched two-scanner payloads are 5,965 B hard shared, 6,217 B Mirror, and 10,419 B independent. Exact replay difference is 0; all serialized payload hashes match; checkpoint tensors roundtrip exactly.
- **Interpretation:** duplication can be avoided for this synthetic shift, but Mirror does not improve upon the cheaper native shared model.
- **Hypothesis:** a useful scanner View may need real sequence/anatomy shifts not explained by calibrated sensitivity maps, and must beat ordinary native conditioning on acquired held-out scanners.

Earlier non-decision-bearing outputs are retained as implementation diagnostics: overdetermined operator (`amendment_01_overdetermined_operator_invalid.json`), masked prior output (`amendment_03_masked_prediction_invalid.json`), incomplete k-space-only image projection (`amendment_04_kspace_output_invalid.json`), and nonuniform serialization framing (`amendment_06_nonuniform_serialization_frame.json`).
