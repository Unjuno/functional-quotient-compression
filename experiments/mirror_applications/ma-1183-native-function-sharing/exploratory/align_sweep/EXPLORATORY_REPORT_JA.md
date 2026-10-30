# Exploratory alignment-continuum sweep: MA-1183 posthoc appendix

**Status: POSTHOC_EXPLORATORY_ONLY.** New five seeds [601,607,613,619,631], 7 fixed interpolation weights ×4 arms = **140 new students**, all 500 AdamW updates. Unlike the original three worlds, the analytical shared and independent teacher weights, inputs and batch indices are *kept identical* within each seed across γ; changing γ is the tested intervention. Fresh/OOD were constructed only after 140 checkpoint SHA256 values were sealed. All 140 Fresh/OOD scores replay exactly.

**The mixing coefficient γ is not the private variance fraction.** It is the weight in the normalized target mixture. A crossover in this specific task is not a universal phase transition in neural networks.

## Fixed-grid results (ReLU, five independent teacher seeds)

| γ | Mirror / FiLM Fresh NMSE (paired geometric mean) | Mirror / coupled-rank1 | Mirror / small independent | Mirror ≥10% better than FiLM |
|---:|---:|---:|---:|---:|
| 0.00 | 0.7047 | 1.0187 | 0.5840 | 4/5 |
| 0.10 | 0.7629 | 0.9924 | 0.6642 | 4/5 |
| 0.20 | 0.9412 | 1.0221 | 0.9253 | 2/5 |
| 0.30 | 1.1267 | 1.0504 | 1.1826 | 0/5 |
| 0.40 | 1.1877 | 1.0379 | 1.2400 | 0/5 |
| 0.60 | 1.2386 | 0.9842 | 1.3941 | 0/5 |
| 1.00 | 1.4092 | 1.0396 | 2.1080 | 0/5 |

**Observation:** Mirror's estimated paired advantage versus FiLM is present at γ=0 and γ=0.1 (ratio below1), uncertain near γ=0.2, and reverses near γ=0.3 and thereafter. This pattern is only a posthoc diagnostic, not an independently preregistered Mirror superiority claim. The coupled rank1 model is almost equal to Mirror throughout the sweep, consistent with exact representation inclusion.

**Uncertainty:** At γ=0.2 the geometric mean Mirror/FiLM ratio is 0.9412, paired log standard error u_c≈0.0743, and a descriptive k=2 interval corresponds to ratios 0.811–1.092; the benefit is genuinely uncertain with five seeds. At γ=0.3 the ratio is 1.1267, u_c≈0.0382, descriptive range 1.044–1.216. Neither interval is a calibrated 95% CI. At γ=0.1 the paired ratio is 0.7629, u_c≈0.1286, interval 0.590–0.987.

**Interpretation:** A plausible conditional design principle is to reserve Mirror for a geometrically coherent shared component and represent the residual with a separate small private mechanism. Next experiment must compare a Mirror+private model with equally budgeted FiLM+private and coupled-rank1+private, not a crippled pure FiLM.

**Verification:** This appendix was planned after inspecting MA-1183 main Fresh; training code/data protocol for this *separate sweep* was committed locally before running new sweep seeds. `SWEEP_TRAIN.log`, `SWEEP_SCORE.log`, `SWEEP_VERIFY.log`, `SEAL.json`, `VERIFICATION.json` included. Main MA-1183 frozen gate remains FAIL 0/5.
