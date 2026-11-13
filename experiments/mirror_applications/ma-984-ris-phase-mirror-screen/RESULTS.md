# MA-984 results — ideal RIS phase-control codes

Status: **FAIL for Mirror-specific attribution.** A two-coordinate phase ramp passes the simulated quality and command-byte gates against full-element 2-bit programming, but an ordinary native phase-gradient controller produces the identical codes, rates, and bytes.

## H — hypothesis

A compact two-coordinate phase profile can preserve useful RIS beamforming while reducing controller command state versus independent per-element phase programming.

## T — execution

**Fact.** The ideal single-user simulator used a 64-element RIS, a direct complex path, and 128 cascaded channels per world with phase ramps plus smooth curvature/noise. Fresh worlds were 98411, 98413, and 98417; development world 98401 was separate. Conditions were continuous per-element alignment, native 2-bit per-element phase control, native 2-bit eight-group control, 5-bit two-coordinate Mirror phase view, and the matched native 5-bit linear phase-gradient controller. The full packed command stream includes a fixed 16-byte decoder/header. Every command was decoded before rate scoring.

**Fact.** Fresh mean rates (bits/s/Hz) were 3.3251 continuous upper, 2.9921 full-element 2-bit, 2.9888 eight-group 2-bit, and 3.3160 for both Mirror and native phase gradient. Mirror retained 99.73% of the continuous upper in each world. Its phase command used 10 bits per scenario and serialized to 176 B for 128 scenarios, versus 2,064 B for full-element 2-bit control (91.5% fewer bytes). Grouped 2-bit control used 272 B but retained only about 89.9% of continuous rate.

**Fact.** The Mirror and native phase-gradient controls decode the same intercept/slope coefficients, reconstruct identical 64-element phase patterns, and produce identical rate sequences in every fresh world. Each still updates all 64 physical RIS elements; the two independent phase coordinates do not mean only two elements are actuated. No energy or switching latency was measured.

## D — decision

**FAIL for Mirror-specific attribution.** The preregistered quality and byte gates pass against native full-element 2-bit control, but the matched native phase-gradient control exactly aliases Mirror at the same 10 command bits and 176 B. This is evidence for a useful ideal low-dimensional phase profile on the selected smooth synthetic channels, not a Mirror-specific advantage or a hardware result.

## C — strongest counter-hypothesis

**Interpretation.** Smooth channel phase profiles are already represented by ordinary phase-gradient control. Real channels with element-level irregularity, multipath interference or quantization/calibration errors may require more coordinates or per-element exceptions.

## U — unresolved

- PA290's joint AP/RIS optimization and multi-user interference.
- Public measured RIS channels and robustness to phase noise, mutual coupling, amplitude loss and OOD users.
- Hardware controller energy, phase-switch settling latency and total control-link overhead.
- Whether private per-element corrections retain quality without erasing the command-state saving.

## Evidence separation

- **Fact:** listed synthetic rates, packed bytes/bits, decoded-pattern equality and 64 physical elements updated.
- **Interpretation:** the low-rank phase function is useful for this smooth channel family, while native linear phase is equally sufficient.
- **Hypothesis:** real multipath channel families may need a larger Mirror code plus rare private corrections; this screen does not test that.
