# MA-405 — StyleGAN-style conditional FFN weight modulation

## H — falsifiable hypothesis

For context-conditioned tasks using one shared two-layer ReLU network, StyleGAN-style input-channel modulation followed by per-output demodulation will recover context-specific behavior at lower actual inference bytes than independent context FFNs, and a four-angle orthogonal Mirror weight view will improve the quality/byte frontier over this native modulation and equal-byte FiLM/rank-one controls.

## T — protocol

This is a synthetic mechanism screen, not a Transformer or StyleGAN reproduction. Each context permutes the input features of a fixed teacher MLP. Methods are shared, native input-channel modulation plus output demodulation, four-angle Givens weight view, FiLM, rank-one context weight residual, and independent context FFNs. Development worlds select learning rate; three held-out worlds are locked before development. Payload bytes include weights and all context codes/generators. Report accuracy, effective-weight diversity, MAC proxy, examples, updates, and wall time.

The goal is to test whether orthogonal geometry adds value beyond the known modulation mechanism, not to claim modulation itself as novel.

## Result

**FAIL.** Fresh accuracy was 57.64% Mirror vs 63.28% StyleGAN-style modulation, 72.37% FiLM, and 89.83% independent FFNs. See `STATUS.md` for H/T/D/C/U and `artifacts/runs.jsonl` for all individual rows.

