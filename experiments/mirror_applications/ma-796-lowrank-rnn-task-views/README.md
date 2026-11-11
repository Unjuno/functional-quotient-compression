# MA-796 — Shared low-rank recurrent dynamics with task Views

Status: **FAIL for Mirror-specific value** (synthetic recurrent-composition evidence is positive within scope)

## Prior-art delta

PA212 motivates shared low-dimensional recurrent components with task dependence at distinct computational loci. This experiment isolates the recurrent transition locus; input and readout remain shared and task-independent. The task coordinate is known from two compositional factors, and two diagonal task combinations are held out from all matrix and basis fitting.

## Fact

Seed 79601; state dimension 8; nine tasks over a 3×3 factor grid; two heldout corners; 256 train sequences for each of seven observed tasks and 256 independent test sequences per task; sequence length 32. The independent full-matrix oracle uses 3,415 B and has zero rollout error. Shared core without task code uses 1,492 B but heldout rollout nRMSE .08093. Mirror recurrent Views use 2,431 B, heldout nRMSE 3.21e-8 and all-task nRMSE 2.66e-8; largest spectral radius is .557. Removing either learned shared recurrent basis raises mean task rollout nRMSE to .0623/.0625 for tasks with the corresponding factor nonzero. Native additive low-rank task conditioning has the same 2,431 B payload hash and identical metrics. Active MAC proxy is 160 per step; reconstructing nine task matrices costs 1,152 MAC. There were zero optimizer updates. CPU wall times are small single-run synthetic calibrations and are not throughput evidence. All inference payloads roundtrip exactly.

## Interpretation

The shared recurrent components support compositional generalization to the two heldout factor combinations in this designed stable linear system, at 28.8% fewer bytes than storing nine independent matrices. Component ablation provides causal evidence that both recurrent bases contribute. This does not establish a Mirror-specific gain: the ordinary native low-rank recurrent conditioning exactly aliases the View in function, payload and compute.

## H / T / D / C / U

- **H:** Compact recurrent task Views over shared low-rank dynamics generalize heldout task compositions with lower complete storage than independent RNN matrices, stable long-horizon behavior, and component-specific causal sensitivity.
- **T:** Seed 79601; synthetic 8D linear RNN; nine two-factor tasks; diagonal corners held out; 7×256 train sequences, 9×256 test sequences, horizon 32. Compared independent oracle matrices, shared core only, Mirror task codes, native additive low-rank control. Measured full NPZ bytes/hash, one-step and long-rollout errors, spectral radius, component ablation, MAC proxies, timing and payload roundtrip; no gradient updates.
- **D:** **FAIL** for Mirror-specific value because native low-rank task conditioning exactly matches the View's payload and function. The scoped compositional mechanism result passes the quality/storage/stability thresholds against independent matrices.
- **C:** The teacher is generated exactly from two additive shared recurrent basis matrices with known task factors; standard low-rank conditional dynamics should recover it, and the native control does.
- **U:** Natural or trained RNNs, inferred task IDs, nonlinear dynamics, task-sequence adaptation, fresh seeds, optimization effects, streaming CPU/GPU deployment and broader task/rank frontiers.

Fresh stayed sealed after exact native alias. No trained-RNN or general task-generalization claim is made.
