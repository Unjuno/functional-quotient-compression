# FQC research agent contract

Read `docs/RESEARCH_STATE.md`, `docs/handoff/LOCAL_RUNBOOK.md` and
`docs/handoff/OPENCODE_PROMPT_JA.md` before local experiments.

- Work from measured files and manifests, not optimistic chat summaries. Do not invent results or completion percentages.
- Preserve `experiments/t282/code/`, `data/`, `locks/`, `records/`, and historical `provenance/t282/` as reference evidence.
  New implementations and runs belong in a new research branch and new directories. Additive corrections must explain provenance.
- Existing user changes must be preserved. Never use `git reset --hard`, `git clean -fd`, force-push or blanket deletion.
- No push, merge, tag, release, public announcement, paid compute or independent model training without explicit approval.
- Never read credentials, `.env`, SSH keys, shell history, unrelated home folders or browser data. Do not bypass denied operations.
- No `curl | sh`, `sudo`, global package changes, pickle fallback, or `trust_remote_code=True` without review.
- Use a local virtual environment. Record versions; do not replace an existing CUDA installation with a CPU wheel.
- CPU is the reproduction reference. MPS is unverified; do not enable silent CPU fallback or fast math and label the result native MPS.
- Checkpoints must match the frozen hash for historical reproduction. Different upstream bytes are a new input revision, not a reason to change the old hash.
- Calibration, development and final test must be disjoint and hash-locked before selection. Old authored probes and the 44 prompt prefixes are already exposed.
- Never choose a candidate or threshold using final-test outcomes. Sample size is not a power guarantee.
- Compare actual final bytes using the same compression and metadata accounting. Separate KL from NLL, document means from token weighting,
  and storage bits from dense FP32 execution. FQC-specific sharing superiority and 64x quality are unresolved.
- Save commands, exit codes, source/input/artifact hashes, all attempted candidates, per-document outcomes, and explicit PASS/FAIL/UNCERTAIN/BLOCKED.
- If a dependency, hash, tokenizer or parity check fails: preserve logs, stop that stage, document the blocker. Do not silently relax tolerance.
- Run a bounded pilot before long work; no uncontrolled candidate sweeps or perpetual background loops. End each run with a resumable status report.

Artifact-free checks: `python -m pytest -q experiments/t282/tests` (149 historical-core checks).
Preflight: `python scripts/fqc_preflight.py --output runs/preflight-001.json`.
Frozen rebuild: `python scripts/fqc_rebuild_frozen.py --models-root ./models --output runs/rebuild-001`.
