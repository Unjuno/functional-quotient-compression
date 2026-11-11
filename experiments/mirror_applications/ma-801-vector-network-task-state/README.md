# MA-801 — Factorized task × state codes over rank-1 atoms

Status: **FAIL for Mirror-specific value and registered byte gate**

## Prior-art boundary

PA213 requires per-input local atom inference over reusable rank-1 weights; PA25 motivates dynamic fast-weight alternatives when input or session state changes a functional code. This screen performs a new top-2 atom selection on every input. Its state code is supplied as an input-regime label, so it does not establish learned fast-weight programming or state inference.

## Fact

Seed 80101; four task factors × four state factors × eight rank-1 atoms; checkerboard half of the 16 task-state pairs held out; 256 examples per pair; top-2 local atom selection; one selection iteration; 28 distinct active atom subsets; zero optimizer updates. Independent full coefficient tensor: 2,275 B, heldout nRMSE 0. Task-only and state-only controls use 1,861/1,863 B and heldout errors 1.038/.800. Factorized task-state views: 2,269 B, heldout nRMSE 1.91e-8. This is only 0.26% fewer bytes than independent, missing the preregistered 25% threshold. The native CP factorization has the same payload hash `583e28e9abb949ab0b08d4e34edc6abd746c952f9911c3297505153a7dc4811e` and exactly the same function. All payloads roundtrip exactly. Active MAC proxy is 80/example; decoding all task-state coefficients costs 256 MAC.

## Interpretation

Task × state factorization recovers unseen combinations in this deliberately separable teacher, while per-input local top-2 selection remains active. Actual compressed bytes show that the factorization barely reduces storage because the independent tensor compresses well and factor metadata consumes most of the theoretical savings. Native CP coefficients exactly account for the result. This is a second consecutive exact native factorization alias in the compositional atom family (MA-800 and MA-801); pause that lane pending a Mirror-distinct design.

## H / T / D / C / U

- **H:** A separable task × state coefficient view over shared atoms generalizes heldout pairs and saves at least 25% complete bytes while retaining local sparse inference.
- **T:** Seed 80101; four task factors, four state factors, eight shared atoms; checkerboard split; 256 examples per pair; compare independent, task-only, state-only, View, and native CP controls. Measure actual NPZ bytes/hash, heldout output error, active atoms and subsets, inference iterations, MACs, example count, wall time, zero updates.
- **D:** **FAIL.** Heldout quality is 1.91e-8 and local inference selects top-2 dynamically, but savings are 0.26%, far below 25%, and Mirror exactly aliases native CP factorization.
- **C:** The teacher coefficient tensor is separable by construction, and ordinary CP factorization recovers the same tensor; serialization overhead plus low entropy erases the nominal parameter savings.
- **U:** Trained Vector Networks, learned task/state inference, true fast-weight programming, realistic OOD distributions, quantization, larger banks, and fused runtime.

Fresh task-state pairs stayed sealed after the frozen storage gate miss and exact native alias. No trained Vector Network or fast-weight claim is made.
