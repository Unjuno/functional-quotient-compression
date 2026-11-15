# MA-1033 development record

## Fact

Development seed 103300 used the locked 20-update, batch-4 Adam configuration. The official test sample was used only for this development screen and was selected after all states were frozen. Fresh audit seeds 103301–103303 remain unaccessed. All train indices are disjoint from MA-1024, MA-1025 and MA-1026.

| Method | Observed class-domain accuracy | Withheld class-domain accuracy | Harmonic mean | Prompt payload bytes | Train wall, s | Audit encoder images/s |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot CLIP | 78.75% | 86.25% | 82.33% | 306 | — | 24.6 |
| Shared coupled prompt | 86.67% | 88.75% | 87.70% | 1,580,632 | 19.6 | 23.8 |
| Mirror product coordinate | 85.00% | 85.00% | 85.00% | 1,580,814 | 21.5 | 23.1 |
| Additive class + domain coordinate | 86.67% | 88.75% | 87.70% | 1,580,806 | 20.5 | 23.7 |
| Independent cell bank | 76.67% | 51.25% | 61.43% | 1,744,542 | 19.4 | 23.9 |

## Interpretation

Mirror trails the byte-near additive control by 3.75 percentage points on held-out cells and the shared coupled prompt by 3.75 points. The independent bank did not behave as a useful upper bound because the held-out target cells had no positive training examples; it is retained as a non-compositional reference, not used as a capacity ceiling. The full-cell bank is only 9.4% larger than Mirror because the shared prompt and projection dominate bytes. This development screen does not select a new configuration; frozen audit settings remain unchanged.

## Hypothesis

The product-form class/domain Givens coordinate is likely to fail the held-out quality gate, while additive code or ordinary shared prompts explain any transfer. Fresh audit is still required to test that direction.

Raw indices, scores, payloads, hashes and timing are in `runs/development_103300.json`.
