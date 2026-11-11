# MA-797 — Factorized task-dependence locus code

Status: **FAIL for Mirror-specific attribution**

## Prior-art delta

PA212 recommends testing task dependence at separate computational loci and validating components through interventions. This synthetic screen places a rank-1 task update at one of three stages (input, recurrent, or readout) and asks whether a compact locus × task-factor address recovers unseen positive-factor tasks.

## Fact

Seed 79701; 8D three-stage linear system; three loci × factors −1/0/+1; all three +1 tasks held out from fitting; 256 test inputs/task; zero optimizer updates. Independent full stage weights use 2,888 B and have zero oracle output error. Shared stages without task codes use 1,211 B and heldout nRMSE .02009. Factorized locus Views use 2,018 B and heldout nRMSE 4.50e-8, a 30.1% reduction versus independent. Applying the view to its correct locus yields nRMSE 4.15e-8; applying the same task factor to a wrong locus yields .02809. Active MAC proxy is 208/example for the rank-1 update path vs 192 for dense independent application; per-task materialization is 576 MAC. The View and native low-rank conditional update have identical payload hash `9dfcfeabb277cc2bb6dd178e17aaab60568b5d4c14f31eedcfbcc3860e8ac414`, bytes, and metrics. All payloads roundtrip exactly.

## Interpretation

The locus code identifies the generating stage in this designed task: the correctly addressed update reproduces heldout outputs and wrong-locus intervention degrades them. However, ordinary native low-rank conditional updates give exactly the same functional family and bytes. This is not Mirror-specific evidence. Rank-1 application adds active compute versus dense independent matrices; the shared-only control is 1 B above the .02 quality threshold.

## H / T / D / C / U

- **H:** A task scalar and input/recurrent/readout locus address recover heldout task functions with at least 25% fewer complete bytes than independent matrices and identify the causal locus.
- **T:** Seed 79701; synthetic 8D three-stage function; −1/0 tasks fit, +1 tasks held out; independent, shared-only, factorized View, and native low-rank control. Actual NPZ bytes/hash, heldout/all-task output nRMSE, correct/wrong-locus interventions, MAC proxies, 2,304 test examples, CPU matrix-chain wall calibration and zero updates.
- **D:** **FAIL** Mirror-specific value: the View saves 30.1% bytes and recovers the heldout outputs, but exactly aliases native low-rank conditioning and adds active MACs. The locus effect itself is established only for this synthetic linear task.
- **C:** A standard conditional rank-1 update plus a locus index already expresses the same mathematical operations.
- **U:** Training-derived locus inference, nonlinear dynamics, natural task quality, fresh seeds, learned routing and fused GPU implementation.

Fresh stayed sealed after exact native alias. Together with MA-796, this triggers a pause on the closely related latent-dynamics task-code lane pending a Mirror-distinct proposal. See the family diagnostic linked from STATUS_BOARD.
