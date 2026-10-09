# MA-596 — Prompt first, Mirror residual second, private residual last

Status: **FROZEN_BEFORE_DEVELOPMENT**. Branch: `research/ma-596-prompt-mirror-private-escalation-20261009`. Prior art: PA117/PA119; integration control: native rank-4 PCA and independent prompts.

## H — Hypothesis

A continual prompt system can preserve each task's held-out continuation quality by starting with a shared prompt, adding a compact rank-4 Mirror coordinate, and storing a private residual only when a fixed validation rule requires it. The experiment measures the quality, bytes, and retention point where private state becomes necessary.

## T — Planned execution

Use frozen Pythia-70M and WikiText-2. Train eight 8-token prompts sequentially per bank on disjoint article task streams. Development uses train articles and seeds 59601/59602. After each task, recompute the shared mean and rank-4 basis, re-encode all seen tasks, and evaluate old-task drift. A task receives a private FP16 prompt residual when its held-out NLL is more than 0.05 nat/token above its independent full-prompt control. Fresh uses unseen test articles and seeds 59611–59613 after development is complete.

Controls: zero prompt, independent prompts, shared mean, native rank-4 PCA residual codes, Mirror rank-4 residual codes, and rule-triggered private escalation. Charge actual NPZ payload bytes for mean, basis, codes, residuals, indices, and metadata. Record prompt training, re-encoding, inference, and virtual-token costs.

## D — Decision

Pending frozen development and fresh results. Mirror-specific attribution requires improvement over the matched native rank-4 coding. If both development banks pass tiered quality and byte gates but exactly alias native coding, run fresh only for the separately preregistered private-escalation frontier.

## C — Strongest counter-hypothesis

The apparent benefit may be ordinary low-rank PCA compression; private residuals may erase its byte savings, and a shared prompt may drift as tasks arrive.

## U — Unconfirmed

Natural language task classification, explicit task routing, a true private LoRA adapter, larger banks, other backbones, and long-horizon continual retention remain untested.

## Evidence labels

Facts, interpretations, and hypotheses will be reported separately in the checked results and verification record.
