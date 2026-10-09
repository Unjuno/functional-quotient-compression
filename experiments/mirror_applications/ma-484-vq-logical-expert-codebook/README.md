# MA-484 — VQ logical expert codebook

Status: **FAIL**.

## H / T / D / C / U

**H:** A shared expert basis plus a learned VQ address should represent 64 useful logical expert functions with low collisions, <=0.05 held-out NRMSE, and <=50% of independent-expert bytes.

**T:** Synthetic 32D linear expert bank, 64 experts, shared base plus four shared residual directions. Development worlds 48400-48401 fit K={8,16,32,64} codebooks; fresh worlds 48410-48412 × seeds 0-2. Compared independent full vectors, shared FP32 low-rank coordinates, generic full-vector VQ, and VQ coordinates over shared basis. Actual serialized torch payloads include all codes, codebooks, basis and metadata.

**D:** FAIL. No Mirror K met quality or collision gates. At K64, Mirror averaged 3,933B but NRMSE .3068 with 86.3% code collisions. The shared FP32 low-rank control represented the synthetic bank exactly at 3,681B. Independent experts were 9,833B. Generic full-vector VQ had even higher errors (>1.2 NRMSE). Thus quantized logical experts lost substantial function quality and were dominated by the simpler shared-basis coordinates.

**C:** The teacher was exactly rank four in a known shared basis, so FP32 coefficients were the natural representation; quantizing them spends bits without saving the basis. The codebook was fitted in the full vector space then projected for Mirror, which may disadvantage the coordinate codebook.

**U:** Trained nonlinear MoE experts, task routing, a codebook fitted directly in coordinate space, and natural expert diversity remain untested. No capacity claim follows from 64 possible labels.
