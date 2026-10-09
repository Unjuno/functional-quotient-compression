# MA-596 — Prompt first, Mirror residual second, private residual last

Status: **FAIL**. Branch: `research/ma-596-prompt-mirror-private-escalation-20261009`. PA117/PA119.

## H — Hypothesis

A continual prompt system can retain held-out continuation quality by starting with a shared prompt, adding rank-4 task coordinates, and storing private prompt residuals only when a fixed validation rule requires them. The preregistered target was ≤0.75× the bytes of independent prompts.

## T — Execution

Frozen Pythia-70M-Deduped; WikiText-2 raw train split; two development seeds (59601/59602); eight article tasks per bank; 8-token prompts and 64 updates per bank. After every task, the shared mean and rank-4 PCA basis were recomputed, all previous tasks re-encoded, and held-out NLL rechecked. Fresh test articles (59611–59613) stayed sealed. Controls were independent prompts, shared mean, rank-4 code, exact native PCA code, and private residual escalation. All prompt/code/residual/index/schema bytes were included in uncompressed NPZ payloads.

## D — Decision

**FAIL.** Rank-4 alone used 42,564 B versus 66,594 B independent prompts (0.639×), but maximum per-task NLL loss was +0.0779/+0.0896 nat/token. Private escalation restored maximum loss to +0.0209/+0.0202, with 1/8 and 2/8 tasks escalated, but payloads grew to 51,282/59,478 B (0.770×/0.893×), missing the ≤0.75× gate in both seeds. The rank-4 representation exactly aliases native PCA.

Full deployment sizes including the common 166,029,852 B Pythia weights were 166,081,134/166,089,330 B for tiered codes versus 166,096,446 B for independent prompts. This saves only 15,312/7,116 full-system bytes because the backbone dominates.

## C — Counter-hypothesis

Ordinary PCA explains compactness; private residuals recover quality but erase the registered storage margin.

## U — Limits

Only small WikiText article prompt banks on one 70M model were studied. No fresh generalization, task router, true private LoRA adapter, larger task count, or optimized inference kernel is established. Fresh data stayed sealed after the byte-gate miss.

Facts, interpretations, and hypotheses are separated in `RESULTS.md`. Verification and exact same-seed payload replay are recorded in `VERIFICATION.json` and `verification_report.json`.
