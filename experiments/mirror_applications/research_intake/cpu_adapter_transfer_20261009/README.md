# CPU real-digit learned LoRA task-code audit — MAT01 / MAT02 (2026-10-09)

**Scientific scope:** Two **negative** pilot studies on one public handwriting image dataset, under distinct development and fresh seed sets. Research only; no status promotion, native LoGo reproduction, natural-LM claim or production GPU performance claim.

- [REPORT.md](REPORT.md): methods, every strong control, full fresh means, decisive failures and important one-forward latency caveat.
- [MAT01 preregistration](MAT01_FROZEN_PROTOCOL.json) committed prior to development/fresh data.
- [MAT02 preregistration](MAT02_FROZEN_PROTOCOL.json) committed **after** MAT01 result and before MAT02 development/fresh; new fresh seeds.
- [Source](source/) and [full raw results](results/): `mat01/{dev,fresh}_raw.csv`, `mat02/{dev,fresh}_raw.csv`, [compact results](results/RESULTS_CORE.csv), [verification hashes](results/VERIFICATION.json) and [same-seed independent replay](results/REPLAY_VERIFICATION.json).
- [REPRODUCE.md](REPRODUCE.md): command lines, hardware, data and leakage firewall.
- [MAT03 correction sketch](MAT03_NEXT_TEST.md): next experiment must use the **same physical input** to make genuine one-pass multi-output claims; rank sweep and matching native heads are required.

Original canonical worker branch and scientific MA-1178 status remain unchanged. The previous CPU batch MA1171/MA1183 also remains unchanged; see its frozen documentation on the parent branch.
