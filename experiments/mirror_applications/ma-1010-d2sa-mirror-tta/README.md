# MA-1010 — D2SA slice-distribution Mirror TTA

A bounded synthetic mechanism screen of compact scanner/volume and slice coordinates under a self-supervised two-stage MRI test-time adaptation objective. This is not an official D2SA reproduction: official source is Apache-2.0, but its README requires pretrained checkpoints, MRI data, and upstream U-Net code that are not present in the execution environment.

The official implementation was inspected at commit `2650d483a7cd77b13242e88719094ee837a45446`. Adaptation equations in this experiment are independently implemented; third-party nanoGPT and official D2SA sources are unmodified.

See `PROTOCOL.json` for frozen conditions and gates. Fresh seeds remain sealed unless the development gate passes.
