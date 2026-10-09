# MA-596 results — development screen

Status: **FAIL**. Fresh test articles were not opened because the tiered method missed the frozen byte gate in both development banks. The 0.05 nat/token private-residual threshold was met by the resulting tiered prompts.

## H — Hypothesis

A continual prompt system can preserve held-out continuation quality by starting with one shared prompt, adding rank-4 task coordinates, then storing a private prompt residual only when the frozen validation criterion requires it. This should use no more than 75% of independent prompt-bank bytes.

## T — Execution

- Frozen model: Pythia-70M-Deduped, revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; model weights SHA-256 `3da388330e4549156d76b58d6d268c63cd005e9336b4f4d2d378421e7b7a33fd`.
- Frozen tokenizer: `tokenizer.json` SHA-256 `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624`.
- Dataset: WikiText-2 raw train split, SHA-256 `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f`; eight seeded article tasks per bank; 8-token prompt, 64 updates per bank, 64 tokens per update window.
- Development seeds: 59601 and 59602. Each prompt was trained on the first 1,024 article tokens; four held-out windows came from tokens 1,024–2,048. Fresh seeds 59611–59613 and test articles remained sealed.
- Controls: independent FP16 prompt bank; shared mean prompt; rank-4 PCA residual code; identical native rank-4 code control; rank-4 code with threshold-triggered FP16 private residual.
- Actual NPZ bytes include prompt mean, basis, codes, private residuals when present, order/index information, and schema. The frozen backbone is reported separately.

## D — Decision

**FAIL.** Rank-4 code alone used 42,564 B versus 66,594 B independent prompts (0.639×), but missed the per-task +0.05 nat/token quality bound in one task for seed 59601 and two tasks for seed 59602. Private escalation restored all-task NLL to within +0.021 nat/token, but required 51,282 B (0.770×) and 59,478 B (0.893×) respectively, missing the frozen ≤0.75× byte gate in both seeds. One and two of eight tasks escalated. The shared-mean prompt was smaller (9,244 B) but exceeded the quality tolerance in both banks.

The rank-4 Mirror representation exactly matches the native PCA representation. Its bytes, coordinates, and decoded prompts are the same; the private residual is separately charged. Therefore this is also a **FAIL for Mirror-specific attribution**.

## C — Strongest counter-hypothesis

Ordinary PCA explains the compact code-only state. The private residual does recover the missed task quality, but its paid bytes erase the preregistered 25% savings target. The result may reflect prompt-bank geometry on these WikiText articles rather than a continual-learning advantage.

## U — Boundaries

Only eight article prompts per bank, two development seeds, one frozen 70M model, and one 8-update prompt schedule were evaluated. No fresh article generalization, class-incremental task routing, true private LoRA weights, larger task count, or optimized serving kernels are established. Because the byte gate failed, no fresh test results are claimed.

## Evidence classes

- **Facts:** both development payloads replay from files with their recorded sizes and hashes; the mean+rank-4 basis/control arrays are identical; private fallback counts are 1/8 and 2/8; tiered per-task maximum NLL deltas are +0.0209/+0.0202 nat/token; tiered payload ratios are 0.770/0.893.
- **Interpretation:** the tiered mechanism identifies a small number of tasks needing private state, but those residuals prevent the registered storage target. Rank-4 prompt compression is native PCA in this implementation.
- **Hypothesis:** longer training, a different article/task distribution, or a genuinely distinct residual coordinate may change the private fraction, but requires a new protocol.
