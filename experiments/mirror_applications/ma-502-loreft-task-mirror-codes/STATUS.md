# MA-502 status

- Status: FAIL
- Branch: `research/ma-502-loreft-task-mirror-codes-20261009`
- Base: `807d8c15`; verification commit `5f8f7947`; development and fresh complete

H: Clustered LoReFT interventions can use compact discrete task codes.

T: 64D synthetic representation tasks, 64 tasks, shared rank 4, 8 modes; dense, shared FP16, and VQ Mirror controls; 3 fresh worlds × 3 seeds.

D: FAIL. Shared FP16: 3,809B / NRMSE .00018. Mirror VQ: 3,677B / .0927. Dense: 18,025B exact. Mirror saved 132B versus FP16 but exceeded .05 error.

C: FP16 coordinates already compress well; VQ quality is inadequate for marginal bytes saved.

U: Pretrained model, downstream quality, label-aware codebook and larger banks. Mode-ID accuracy is not used because codebook labels are permutation invariant.
