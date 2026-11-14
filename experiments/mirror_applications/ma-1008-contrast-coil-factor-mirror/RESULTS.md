# MA-1008 — Contrast × coil factorized MRI Mirror prior

Status: **FAIL (development gate)**  
Evidence lane: **synthetic mechanism / serialized state / CPU compute**  
Branch: `research/ma-1008-contrast-coil-factor-mirror-20261114`

## H — Hypothesis

A five-coordinate additive chart (global Givens offset + two contrast + two coil coordinates) fitted on eight cells of a synthetic 3×3 contrast × coil grid would predict the held-out (2,2) cell better than hard sharing and an equally small native scalar chart, while using less actual serialized storage than eight independent pair Givens codes.

## T — Test

PA306 motivates continuous contrast conditioning; PA302 motivates preserving explicit multicoil acquisition physics. This CPU screen uses analytic 32×32 complex phantoms, four smooth coil maps, and a fixed 4× Cartesian mask. A small residual CNN is pretrained jointly on 32 examples spanning the eight observed cells, with 240 updates. For each observed cell, a direct scalar or Givens coordinate is fitted on four support cases (80 updates per cell); only those eight fits are projected by least squares onto the five-coordinate additive chart. Cell (2,2) is absent from chart fitting. Eight disjoint cases score development quality. Separate target-support scalar/Givens and full-prior fine-tune are upper controls, not transfer methods. The MRI SENSE adjoint input and acquired-line residual use the same explicit sensitivity maps, mask, and centered FFT in every method.

- Source fit: 32 examples, 240 updates, 0.267 s.
- Observed-cell code fits: 640 updates per code family, 0.700 s Givens and 0.559 s scalar.
- Target upper controls: 4 examples, 80 updates each.
- Inference: approximately 0.0005–0.0007 s/case on CPU; CNN proxy 1,769,472 MAC/case, plus explicit four-coil FFT operations.
- Data are generated synthetic phantoms; no fastMRI acquisition or clinical data were accessed.

## Development results

| Method | NMSE ↓ | PSNR (dB) ↑ | Measured-coil residual ↓ | Actual payload (B) |
|---|---:|---:|---:|---:|
| Hard shared, frozen | 0.021697 | 34.444 | 0.07465 | 9877 |
| Factorized scalar chart | 0.021537 | 34.475 | 0.07556 | 10193 |
| Factorized Givens Mirror | 0.021623 | 34.457 | 0.07489 | 10193 |
| Target-pair Givens code (support upper) | 0.021624 | 34.457 | 0.07496 | 10129 |
| Target full-prior fine-tune upper | 0.015390 | 35.953 | 0.06540 | 9877 |

Observed-cell audit: projected Givens chart mean NMSE 0.036197, versus direct fitted Givens codes 0.036202; projected scalar chart 0.036161, direct scalar codes 0.036165. The chart retained known-cell quality closely in this small screen, but that did not translate into a useful held-out-cell advantage.

Actual serialization comparison uses the same shared model state, operator/architecture metadata, and `torch.save` container: eight direct float32 Givens codes = 10129 B; five factorized coordinates = 10129 B. Measured full-payload saving is 0.000%, below the preregistered 20% gate. Coordinate count alone does not support a storage claim.

## D — Decision

**FAIL at the bounded synthetic development gate.** Mirror is 0.40% worse in NMSE than the factorized scalar control and 0.34% worse than hard sharing; it is 40.5% worse than target full fine-tuning by NMSE. It does not meet the 5% advantage or 10%-of-upper quality gates. Full serialized factorized payloads are tied with the eight-code bank at 10,129 B in the matched bank comparison. Fresh seeds remain sealed.

## C — Strongest counter-hypothesis

The useful contrast/coil effects are either already encoded in the shared prior and measured acquisition input, or can be captured by ordinary scalar conditioning. The Givens coordinate adds no useful held-out interaction and costs training updates to fit its code chart.

## U — Unconfirmed

No acquired MRI, fastMRI benchmark, scanner shift, anatomy preservation audit, or fresh replication. This is not clinical evidence and cannot establish PA306/VarNet performance. The result only falsifies the registered chart under this synthetic model/split.

## Fact / Interpretation / Hypothesis

- **Fact:** on eight held-out synthetic cases, Mirror NMSE is 0.021623; factorized scalar is 0.021537; hard sharing is 0.021697. Full payload bank sizes are equal at 10129 B.
- **Interpretation:** this screen shows no Mirror-specific quality or full-payload benefit over the registered controls; measured code-count reduction was absorbed by serialization/container granularity at this small scale.
- **Hypothesis:** a larger real contrast/coil grid might amortize metadata and make factorization cheaper, but it would still need to beat native continuous conditioning and demonstrate held-out acquired-data fidelity.

The initial target-support screen is preserved separately as superseded exploratory evidence: `runs/amendment_00_exploratory_superseded.json`. Amendment 01 fixed the target-cell leakage before its metrics and is the decision-bearing run.
