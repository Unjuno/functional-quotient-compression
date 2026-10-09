# MA-501 status

- Status: FAIL
- Branch: `research/ma-501-loreft-mirror-coordinate-20261009`
- Base: `0ea7c36c`; verification commit `d38b911d`; A1 corrects non-contiguous basis storage
- Fresh: 50120-50122 × seeds 0-2

H: Shared representation basis plus Mirror coordinates compress task interventions at <=.05 NRMSE.

T: Synthetic 64D frozen representation vectors; dense, FP16 LoReFT, shared-only and VQ Mirror K8/16/32/64.

D: FAIL. FP16 LoReFT was 3,809B (21.1% dense) at NRMSE .0595, just above .05. VQ Mirror error .466-.701. Dense 18,025B was exact.

C: FP16 coefficients outperform coarse VQ coordinates; discrete codes are poorly matched to continuous Gaussian task variation.

U: Pretrained Transformer, task-level quality and private residual improvements.

A0 non-contiguous basis payloads are exploratory/excluded; A1 materializes rank-4 storage contiguously.
