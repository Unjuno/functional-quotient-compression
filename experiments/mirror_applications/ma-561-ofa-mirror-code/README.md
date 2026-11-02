# MA-561 — OFA architecture code + Mirror function code

Status: SCREENING. Frozen protocol: [`PROTOCOL.json`](PROTOCOL.json). Prior art PA109 establishes Once-for-All subnet specialization by architecture coordinates; this screen asks only whether an additional compact function code adds useful held-out configuration behavior.

The environment has CPU-only PyTorch and no checked-out OFA implementation or accelerator. This experiment is therefore a synthetic OFA-like mechanism screen, not an OFA/ImageNet result. It uses a shared width/depth/kernel-elastic residual network and compares native subnet selection, a generic rank-one output correction, a compact Givens code, and independent configuration maps. The teacher is deliberately an aligned smooth Givens orbit, so even a positive result is an upper-favorable mechanism test.

An A1 amendment replaces an invalid initial development linear-regression upper control with a paid, oracle-calibrated full effective output map per configuration over the shared backbone; the initial development artifacts are preserved. No fresh data had been accessed when this correction was made. No optimizer updates are used. Two development worlds and three fresh worlds are fixed in `PROTOCOL.json`; four joint configurations are held out in every world. All shared maps and method-specific inference state are charged using actual serialized NPZ bytes. A FAIL here only falsifies the frozen synthetic configuration-code hypothesis.

## A1 fresh result

The corrected full-map control is an oracle upper reference and reproduces the target at fresh held-out configurations. Across fresh worlds, Mirror mean/max NRMSE was 7.47e-17/2.10e-16 versus 0.3703/0.9292 for native elastic subnet selection; the generic rank-one residual was 0.3599/0.8363. Mirror's complete serialized payload averaged 4,247 B versus 12,540 B for independent full effective maps (33.9%); rank-one used 4,804 B and did not approach Mirror quality. Native shared-only payload was 4,027 B. Mirror's measured CPU batch time was 0.0379 ms vs 0.0256 ms native, so compute regressed; measurements are short/noisy and CPU-only.

**Disposition: PROMISING only for this deliberately Givens-aligned synthetic mechanism.** This does not establish an OFA result, natural-task benefit, or capacity increase. The initial invalid A0 upper reference is retained in `runs/a0_development`; A1 corrected it before any fresh access.
