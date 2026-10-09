# MA-320 — Tucker expert bank + Mirror expert coefficients

Status: SCREENING. Prior art: PA35 Tucker matrix bank.

## H

A shared Tucker bank with small per-expert Mirror coordinates may reduce logical expert storage while retaining routed outputs. Free Tucker coefficients and generic PCA are mandatory controls.

## T

Synthetic MoE with 128 logical experts, K=8 shared matrices of 16×16, 32 tokens/expert, and deterministic balanced top-1 dispatch. Compare independent full matrices, free Tucker coefficients, Mirror Givens codes, and generic rank-2 PCA. Test aligned and independent expert coefficients. Development worlds 32000–32001; fresh 32010–32012; three seeds each. Every payload charges shared bank and metadata; measure one-time materialization MACs and expert forward MACs.

## D

Pending development/fresh runs.

## C

The aligned experts are generated from the same Mirror orbit; PCA may match, and independent experts may require free coefficients.

## U

Synthetic linear experts only; no learned router, expert training or language quality.

Development: for aligned 128-expert tasks, Mirror payload is 4,555 B / routed NRMSE 5.09e-4 vs free Tucker 6,313 B / 2.94e-4 (28% fewer bytes). Generic PCA is 4,867 B / 3.69e-4, slightly larger and more accurate; its fit is ~0.68 ms vs Mirror ~5.5 ms. All share one-time materialization cost of 262,144 MACs; the selected expert forward is 256 MACs/token. Independent coefficients are exact under free Tucker, while Mirror NRMSE is ~1.10. Fresh checks frozen settings.
