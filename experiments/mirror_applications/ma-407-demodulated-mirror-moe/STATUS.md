# MA-407 status: FAIL

- Branch: `research/ma-407-demodulated-mirror-moe-20261008`
- Protocol frozen: `2a1a6f6`
- Development complete: yes; all methods selected LR 0.01
- Fresh complete: yes; worlds 40710–40712 × 3 seeds
- Payload replay: verified for all 126 records

## H — falsifiable hypothesis

Row-norm demodulation of a context-addressed Givens expert would improve fresh accuracy by at least 1pp and reduce activation RMS coefficient of variation by at least 25% versus identical raw Givens views.

## T — executed

Four-context synthetic classification with a fixed shared MLP teacher and four context Givens views. Compared shared, raw Givens, Givens plus row-norm demodulation, IA3, rank-one context residual, and independent experts. 200 AdamW updates × batch 128; 2 development worlds × 3 seeds and 3 fresh worlds × 3 seeds. Actual serialized PyTorch inference payload bytes retained. CPU PyTorch 2.14.1.

## D — FAIL

Fresh means: shared 58.50% accuracy / 9,829B; Mirror raw 66.45% / 10,081B; Mirror demodulated 66.45% / 10,081B; IA3 60.18% / 10,529B; rank-one 60.74% / 11,485B; independent experts 59.79% / 34,021B. Mirror raw and demodulated activation RMS CV both 0.0093, exactly equal across corresponding fresh runs. Their active MAC proxy is 5,568/example vs IA3's 5,568; mean train wall time was 0.352s raw and 0.608s demodulated.

## C — strongest counter-hypothesis

The Givens rotations are orthogonal, so the per-output hidden weight-vector norms being normalized are invariant under the input-plane rotation. Demodulation therefore has no functional effect in this parameterization. This is a structural null, not evidence that demodulation fails for non-orthogonal scale-producing codes.

## U — unresolved

Non-orthogonal Mirror codes, learned routing, real MoE/Transformer workloads, throughput, and quality near convergence remain untested. Independent control underperformed here due limited training and this teacher setup; it is not an upper bound in this screen.
