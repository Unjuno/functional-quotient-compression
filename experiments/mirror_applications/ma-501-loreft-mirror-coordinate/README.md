# MA-501 — LoReFT subspace + Mirror coordinate

Status: **FAIL** for the registered <=.05 NRMSE gate.

**H:** A shared rank-4 hidden representation basis plus per-task coordinates stores interventions below half the bytes of dense vectors at matched quality.

**T:** Synthetic frozen 64D representation interventions, 64 tasks, shared rank 4. Development worlds 50100-50101 fitted PCA basis and VQ codebooks. Fresh A1 worlds 50120-50122 × seeds 0-2 compared dense vectors, FP16 shared LoReFT coefficients, no-coordinate shared basis, and VQ Mirror coordinates K={8,16,32,64}. A0 non-contiguous SVD-view artifacts are preserved and excluded.

**D:** FAIL on quality. FP16 LoReFT coordinates used 3,809B (21.1% dense) but mean NRMSE .0595, narrowly missing .05. VQ Mirror used 3,677-4,573B but error .466-.701. The shared-only model had NRMSE 1.002. Dense storage was 18,025B with zero reconstruction error.

**C:** Continuous FP16 coefficients are already a compact strong control; coarse learned VQ adds large distortion and no storage advantage over FP16 coefficients at comparable quality. Gaussian continuous task coordinates are unfavorable for a small discrete codebook.

**U:** Learned pretrained Transformer interventions, task-level downstream quality, private residual allocation, and optimizing the basis for the task distribution remain untested.
