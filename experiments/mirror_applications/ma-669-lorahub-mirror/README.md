# MA-669 — LoRAHub coefficients from Mirror task code

## H — hypothesis

A development-fitted map from few-shot support statistics to adapter coefficients may reduce coefficient-search work while preserving heldout quality. It must match direct coefficient fitting and stay within LoRAHub's payload budget.

## T — execution

Synthetic linear regression with four-dimensional inputs/outputs and six shared rank-1 experts. Development seed 66901 fit the linear support-statistic-to-code map on 12 task identities. Fresh worlds 66911–66913 each used 12 disjoint tasks, 32 support and 128 query examples/task. Compared the fitted Mirror code map, direct least-squares coefficient solve, LoRAHub-style 1,024-candidate gradient-free random search, and independent task deltas. Noise SD was 0.03. No optimizer updates. Actual NPZ payload bytes and objective evaluations were measured.

## D — FAIL

Across 36 fresh tasks, mean query MSE was 0.02644 for the Mirror support map, 0.02612 for LoRAHub search, 0.0000496 for direct least squares, and zero for the stored full-delta reference. Mirror missed the frozen 1.05× direct-fit quality gate. It used one mapped code evaluation versus 1,024 LoRAHub candidates, but its payload was 1,906 B versus 1,220 B for LoRAHub and direct fitting (56% larger), missing the <=5% byte margin. The fixed Pareto gate failed.

## C — strongest counter-hypothesis

The learned support-statistic map does not generalize the sparse task coefficients well enough. Direct least squares solves the small linear problem with one fit and avoids storing a development-fitted map; random search also achieves similar quality with less deployed state.

## U — unconfirmed

No pretrained LoRA bank, language model, task shift beyond this synthetic expert span, or end-to-end few-shot adaptation was tested. The random search is a fixed small-budget LoRAHub analogue, not a reproduced published implementation.

## Fact / Interpretation / Hypothesis

- **Fact:** Mirror MSE 0.02644, direct solve 0.0000496, LoRAHub-style search 0.02612; payloads 1,906 B / 1,220 B / 1,220 B.
- **Interpretation:** fewer coefficient evaluations did not buy matched task quality and added stored state.
- **Hypothesis:** a more expressive code map might help larger expert banks, but must beat direct regression and account for its mapping bytes.
