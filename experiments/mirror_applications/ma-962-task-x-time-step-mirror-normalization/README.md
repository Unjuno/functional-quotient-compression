# MA-962 — Task × time-step Mirror normalization View

Status: **FAIL (development Mirror-specific gate); audit unopened**
Branch: `research/ma-962-task-time-mirror-tebn-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Draw 15 selected MA-962 uniformly from 553 eligible P0/UNTESTED rows. The ordered pool, exclusions, cryptographic seed, replayable index and hash are under `source/`.

## H — Falsifiable hypothesis

For four task-specific temporal encodings of MNIST, a shared SNN with one learned temporal-gain profile and one cyclic phase coordinate per task will stay within 1.5 percentage points of task×time TEBN accuracy using at least 2% fewer complete serialized inference bytes, and beat an equal-byte rank-1 task-scalar gain control by at least 1 percentage point.

## T — Execution

A one-hidden-layer leaky integrate-and-fire SNN (784/128/10, 8 timesteps, fast-sigmoid surrogate) was trained on four cyclic shifts of the same rate-coded MNIST task. Conditions were shared temporal gain, native task×time TEBN, cyclic-phase Mirror, equal-size rank-1 task-scalar gate, and independent per-task SNN upper control. Each shared condition used 800 AdamW updates per world; each independent task used 800 updates. Two development seeds (96201, 96202), 50,000 MNIST training images, and 2,048 fixed development images per seed across the four tasks. No post-freeze tuning occurred.

Only the official MNIST training IDX files were acquired. Their compressed SHA-256 values and split boundaries are recorded in `source/data/manifest.json` and `source/development_summary.json`. The official test files were **not downloaded, decoded, or opened** because the Mirror-specific development gate failed.

## D — FAIL

- **Quality/storage gate passed in both seeds.** Mirror accuracy was 91.64% and 91.15%; task×time TEBN was 91.58% and 91.08%, within the 1.5-point limit. The complete serialized payload was 414,436 B versus 426,532 B for TEBN (0.9716×; 2.84% fewer bytes), passing the predeclared 0.98× gate.
- **Mirror-specific gate failed in both seeds.** The equal-byte rank-1 task-scalar control reached 91.93% and 91.36%. Mirror was lower by 0.293 and 0.208 percentage points; the gate required a 1-point Mirror advantage.
- Mean accuracy across the two development worlds: Mirror 91.3940%, native task×time TEBN 91.3269%, rank-1 scalar gate 91.6443%, shared temporal gain 91.3879%, independent per-task SNN 91.6809%.
- Mirror payload was 414,436 B, shared temporal-gain baseline 414,162 B, rank-1 scalar gate 414,888 B, TEBN 426,532 B, and independent four-model upper control 1,652,450 B. Complete model bytes make the Mirror saving over TEBN modest because synaptic weights dominate.
- Total wall time for all ten fixed-budget runs was 117.6 s on single-thread CPU. Active MAC proxy was 813,056/example; per-run updates, training time, spike counts, throughput and payload hashes are in `RESULTS_CORE.csv` and the JSON summary.
- All ten serialized inference payloads passed size/hash checks and strict state reload. Four pre-data model tests and the draw replay passed.

## C — Strongest counter-hypothesis

A simple scalar task gain on the shared temporal profile captures at least as much useful specialization as a learned phase shift. It used nearly the same bytes and was more accurate in both worlds. The Mirror phase coordinate did not establish an incremental functional benefit over this native gate.

## U — Still unknown

Generalization on the locked official MNIST test set was not measured. Other task families, temporal precision sweeps, near-convergence capacity, neuromorphic hardware latency and energy remain untested. This two-world fixed-budget screen is not a capacity or hardware claim.

## Fact / Interpretation / Hypothesis

- **Fact:** The measured Mirror payload was 2.84% smaller than task×time TEBN and its accuracy was within 1.5 points, but it underperformed the equal-byte scalar gate in both seeds; audit data remained unopened.
- **Interpretation:** The storage reduction came from compressing a small normalization table while shared synaptic weights dominated total bytes. The observed task-phase variation was handled at least as well by a simpler scalar gate.
- **Hypothesis:** A phase View may help where task-specific temporal profiles are actual shifted copies, but this run does not establish that advantage over a simple gain control.
