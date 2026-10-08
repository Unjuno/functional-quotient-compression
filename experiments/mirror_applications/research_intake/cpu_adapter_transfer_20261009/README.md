# CPU real-digit learned LoRA task-code audit — MAT01 / MAT02 (2026-10-09)

**Scientific scope:** Two **negative** pilot studies on one public handwriting image dataset, under distinct development and fresh seed sets. Research only; no status promotion, native LoGo reproduction, natural-LM claim or production GPU performance claim.

- [REPORT.md](REPORT.md): methods, every strong control, full fresh means, decisive failures and important one-forward latency caveat.
- [MAT01 preregistration](MAT01_FROZEN_PROTOCOL.json) committed prior to development/fresh data.
- [MAT02 preregistration](MAT02_FROZEN_PROTOCOL.json) committed **after** MAT01 result and before MAT02 development/fresh; new fresh seeds.
- [Source](source/) and [full raw results](results/): `mat01/{dev,fresh}_raw.csv`, `mat02/{dev,fresh}_raw.csv`, [compact results](results/RESULTS_CORE.csv), [verification hashes](results/VERIFICATION.json) and [same-seed independent replay](results/REPLAY_VERIFICATION.json).
- [REPRODUCE.md](REPRODUCE.md): command lines, hardware, data and leakage firewall.
- [MAT03 correction sketch](MAT03_NEXT_TEST.md): next experiment must use the **same physical input** to make genuine one-pass multi-output claims; rank sweep and matching native heads are required.

Original canonical worker branch and scientific MA-1178 status remain unchanged. The previous CPU batch MA1171/MA1183 also remains unchanged; see its frozen documentation on the parent branch.

## MAT03 / MAT03R: same original image, one real trunk forward, five useful outputs

- [Initial source/benchmark and source-frozen replication](mat03/README.md): 3 dev seeds and initial 5 fresh plus **5 separately frozen new fresh** 401..405; nine fair native/ordinary/Mirror controls, K=5 binary outputs on the same input, one trunk call for each method.
- [Full report and contradictory evidence](mat03/REPORT.md): initial 5/5 Mirror4 beats same-coefficient linear4; new-seed replication 4/5 but preregistered >=0.01 nat/label margin passes 1/5 only. Native one-trunk five-head is more accurate and faster, with <2% whole-model storage overhead. **No Mirror-specific Pareto win.**
- [All source and raw evidence](mat03/results/), [exact replay](mat03/results/REPLAY_VERIFICATION.json), [portable audit](mat03/check_mat03.py). MAT03 initial protocol is [here](MAT03_FROZEN_PROTOCOL.json) and true frozen-code replication [here](mat03/MAT03R_FROZEN_PROTOCOL.json).
- All scientific registries and the canonical worker branch remain untouched. MAT03 describes the original small output-multiplexing hypothesis, **not** a finished original MA-1175/MA-1178 experiment.
