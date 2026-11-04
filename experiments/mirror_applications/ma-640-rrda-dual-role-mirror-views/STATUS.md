# MA-640 status

- Status: FAIL
- Branch: `research/ma-640-rrda-dual-role-mirror-views-20261104`
- Base commit: `44740ab8`
- Protocol frozen before development: yes
- Fresh/audit opened: no (failure gate reached on development)

## H / T / D / C / U

**H:** A shared adapter basis with role-specific Mirror coordinates represents RRDA edit and suppress functions with less edit-state storage when role updates align; increasing role angle reveals when another basis direction is needed.

**T:** Synthetic 16D two-role outputs at 0/15/30/60/90 degrees; two development seeds (64001, 64002); independent full role adapters, shared rank-1/rank-2 Mirror codes, native rank-1 gate and native shared rank-2 low-rank controls. Same base executor and role schema; measured actual compressed `.npz` inference payload bytes, role-state bytes, per-role nRMSE, operation proxy and wall time. Fresh seeds 64011–64013 remained sealed after the frozen failure gate. No optimizer updates were needed for this deterministic representational test.

**D:** FAIL. On both seeds at aligned 0 degrees, independent adapters were 741 role-state bytes and 1,955 full bytes; rank-1 Mirror was 896 role-state bytes and 2,110 full bytes with zero error. The native gate had 895 role-state bytes and 2,109 full bytes with zero error. At 15/30/60/90 degrees, rank-1 Mirror errors rose from 0.130526 to 1.0; rank-2 Mirror restored zero error but cost 969–970 role-state bytes and 2,183–2,184 full bytes, exactly matching native shared low-rank rank-2 in errors and bytes. Rank-1 uses a lower operation proxy (18 vs 32) but has lower quality as role directions diverge. No Mirror-specific gain observed.

**C:** The two roles are ordinary vectors; a shared basis plus role coordinates is just gating/low-rank factorization, and the compressed container overhead exceeds the tiny independent state in this toy task.

**U:** This does not establish RRDA performance on a pretrained LM, routed prompt locality, or behavior with trained adapters. Development representation results suffice to fail the prespecified storage and Mirror-specific gates; fresh performance is not established.

## Fact / Interpretation / Hypothesis

- **Fact:** All listed byte counts and nRMSE values come from development seeds 64001 and 64002. Native and Mirror rank-2 outputs and serialized sizes are identical. All three tests pass; metric replay has zero difference.
- **Interpretation:** For this two-role adapter construction, Mirror coordinates do not beat standard gating or low-rank controls and do not reduce serialized storage versus independent adapters.
- **Hypothesis:** Larger, trained RRDA adapters might have a different storage crossover, but would still need to beat native factorization at matched bytes and quality to support a Mirror-specific claim.
