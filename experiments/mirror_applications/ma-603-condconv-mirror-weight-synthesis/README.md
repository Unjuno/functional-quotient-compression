# MA-603 — CondConv-style Mirror weight synthesis

Status: SCREENING → development mechanism screen
Evidence lane: MECHANISM / STORAGE / COMPUTE
Base commit: `8fd9f119` (worker-ready baseline plus verified MA-602 result)
Prior art: PA121, CondConv

## Hypothesis

H: On held-out inputs for a non-convolutional FFN whose effective weights vary with input, adding an input-conditioned structured Givens View to a shared expert basis reduces MSE enough to improve the serialized-byte/quality frontier over native linear CondConv coefficient mixing and a parameter-matched ordinary nonlinear coefficient generator.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m(x)` as input-conditioned Givens angles on the expert coefficient vector before shared FFN basis synthesis, so one physical FFN basis can express sample-specific logical FFN weights without storing a separate FFN per input regime.

Native method: CondConv linear combination of shared FFN expert weights. Mirror coordinate: two input-generated Givens angles on four expert coefficients. The ordinary control is a compact nonlinear coefficient MLP. The independent upper reference is the teacher's own fixed basis and generator and is an oracle reference, not a trained student.

## Gates

PASS requires the Mirror to reduce fresh-world held-out MSE by >=10% versus both native linear CondConv and the parameter-matched nonlinear coefficient MLP, while its actual serialized payload is <=1.25x the better control and its synthesis FLOP proxy <=1.25x. FAIL if it misses any of these after development gate; fresh remains sealed. This mechanism screen does not establish language-model capacity.

## Fixed setup

See `PROTOCOL.json`. Development worlds 60301 and 60302; fresh worlds 60311 and 60312 only if both development gates pass. Inputs are generated from independent fixed seeds. Train/dev/fresh examples are disjoint.

## H / T / D / C / U

**H — hypothesis:** adding an input-conditioned Givens coordinate over shared FFN basis coefficients improves the held-out quality/serialized-byte/compute frontier over native CondConv and a compact ordinary nonlinear coefficient generator.

**T — execution:** CPU-only PyTorch 2.14.1; 12→8→8 FFN with four shared basis FFNs; two independently seeded development worlds (60301, 60302), 4,096 train and 1,024 held-out examples, 1,200 AdamW updates per learned method. Compared static shared FFN, native linear CondConv, the two-angle Givens Mirror, two-layer MLP coefficient generator, and a privileged teacher oracle. Actual `torch.save` inference payloads, coefficient-generation/synthesis/FFN MAC proxy, updates, examples, and isolated training wall time are recorded. Fresh 60311/60312 were not opened because the development gate failed.

**D — FAIL:** Mirror MSE was 0.000281/0.001127, versus CondConv 0.000232/0.000989 and nonlinear MLP 0.0000986/0.000360. Thus Mirror was worse in both worlds than both controls. Its 6,221 B payload exceeded CondConv's 5,653 B by 10.0% and MLP's 6,157 B by 1.0%; its compute proxy (1,180) exceeded CondConv (1,120) and MLP (1,088). The predeclared PASS gate was missed; fresh remained sealed.

**C — strongest counter-hypothesis:** the teacher's nonlinear coefficient map is naturally captured by the ordinary small MLP gate; the extra orthogonal coefficient rotations add parameters and synthesis work but distort the useful positive mixture, while native CondConv already captures most of the signal.

**U — boundaries:** this is a fixed-update synthetic mechanism screen with one teacher family, not a natural Transformer or near-convergence capacity result. The teacher oracle has privileged target generator weights and is not a learned student.

## Facts / interpretation / hypothesis

- **Fact:** all 10 saved payloads match their recorded byte sizes and SHA-256 hashes; reloaded outputs had maximum absolute error 0.0. Two tests pass. Both development worlds missed the frozen quality gate.
- **Interpretation:** on this fixture, ordinary nonlinear coefficient generation is a better use of a similar payload budget than the selected Givens View. Native linear CondConv also dominates Mirror on quality, bytes, compute proxy, and measured training time.
- **Hypothesis:** for dynamic FFN synthesis, structured Mirror coefficients may only help when the target coefficient geometry is specifically aligned with the View; this randomized teacher did not provide evidence for such an advantage.
