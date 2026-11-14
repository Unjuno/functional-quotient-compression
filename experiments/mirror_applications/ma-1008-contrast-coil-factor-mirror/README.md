# MA-1008 — Contrast x coil factorized MRI Mirror prior

Status: **FAIL** (bounded synthetic development gate)  
Evidence lane: synthetic mechanism / serialized state / CPU compute.  
Prior art: PA306 multi-contrast MRI INR; PA302 VarNet.

## H — Hypothesis

A five-coordinate additive contrast-plus-coil Givens chart fitted from eight cells of a synthetic 3x3 grid will predict the withheld cell better than hard sharing and an equally small native scalar chart, while reducing actual serialized payload versus eight direct pair codes.

## T — Test

Amendment 01 withholds cell (2,2) from fitting. A shared residual CNN is pretrained on 32 examples from the other eight contrast-coil cells. Eight direct scalar or Givens codes are fit using four support examples per cell, then projected to a five-coordinate chart. The held-out cell uses eight development phantoms. Target-support direct code and full-prior fine-tuning are separate upper controls. All methods use the same explicit 4-coil SENSE adjoint, sensitivities, Cartesian mask, and centered FFT path.

The initial support-adapted screen is retained as superseded exploratory evidence. It did not test unseen-cell transfer. Amendment 01 is decision-bearing. No fresh data were opened.

## D — Decision

**FAIL.** Held-out NMSE is .021623 for Givens Mirror, .021537 for factorized scalar, and .021697 for hard sharing; target full-prior fine-tuning reaches .015390. Matched actual serialized payload is 10,129 B for both the direct eight-code bank and the five-coordinate factorization: no byte saving. The registered quality and storage gates fail.

## C / U

The shared prior and ordinary scalar conditioning may already capture the useful variation. This CPU synthetic screen does not establish acquired MRI, fastMRI, scanner, anatomy-fidelity, or clinical performance. Fresh seeds remain sealed.

Full protocol, metrics, H/T/D/C/U report, and limitations: [RESULTS.md](RESULTS.md).
