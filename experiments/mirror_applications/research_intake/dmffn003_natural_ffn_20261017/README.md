# DMFFN-003 — Natural digit learned FFN transfer and Dual-Input Mirror compression

**STATUS: EXPERIMENTS EXECUTED; native same-byte and heldout generalization gates FAIL.** 2026-10-17.

**Mirror insertion:** one shared SwiGLU has two distinct input projections (gate and up) plus a shared output projection. Learned low-description Givens addresses for gate and up are combined across a 3×3 grid of logical binary classifiers. Unlike previous DMFFN-001, no Mirror-generated teachers are used.

Four **separately frozen, prospectively evaluated** studies use sklearn's 1,797 *real handwritten digits* with 9 manually constructed binary conjunction tasks:
- **A (36 student models, 1,800 steps):** first train nine mutually independent SwiGLU teachers on natural digit classification labels, then distill *only six* task combinations into students, evaluating the three untouched combinations.
- **B (36 models, 2,400 steps):** direct natural-label supervised training for six task combinations and evaluation on three never trained combinations.
- **C (36 models, 2,400 steps):** directly supervise all nine combinations to test **known-task compression**, not unseen combination transfer.
- **D (12 models, 6,000 steps):** longer training in two new split worlds, monitoring overfitting. Every study uses matched shared initializations and minibatch streams for the relevant comparable methods; different split worlds all come from one digit dataset and are not independent corpora.

**Key result C, held-out images, median of 3 split-world means:** Mirror 0.08966 binary cross-entropy (BCE) at 4,270 parameters; equal-parameter native gain 0.09127; **plain shared-FFN with per-task readout 0.03990** at 4,454 parameters; independent nine FFNs 0.03199 at 37,737 parameters. The per-task native readout is a strong and nearly byte-equal winner over Mirror. Actual model tensor/config serialized bytes: Mirror 20,489 B, readout 20,817 B, independent 153,685 B. A full trained input-normalization payload (2,917 B) is charged to each: **23,406 / 23,734 / 156,602 B** respectively. Mirror saves **85.05% full inference state bytes** versus independent, but **does not match its quality**.

**Key result A/B:** Heldout 3-pair BCE: A Mirror 2.489, gain 3.248, **hard-tied 0.684**; B Mirror 3.092, gain 3.860, **hard-tied 0.687**. A narrowly passes its gate only against gain (2/3 worlds), but FAILS to provide useful heldout quality even against hard tying (0/3 worlds). B's broader gate FAILS 0/3. All nine labels are supervised in study C, so C's high accuracy cannot be cited as unseen combination generalization.

**Long training:** mean test BCE (900 → 6,000 updates) Mirror **0.08552 → 0.12778**, gain 0.10092 → 0.15866, independent 0.03797 → 0.04845. In all 12 trial models 900 steps was the best validation checkpoint among measured updates. Extra training alone worsened test generalization on this small dataset.

**Audit:** 18 unit tests, all 120 learned-student inference payloads restored and checked against stored SHA256, actual bytes and complete test metrics; four fresh student deterministic re-trainings reproduced exactly plus one independent-teacher parameter replay. Four protocol/source locks checked; fourteen true serialized, train-only normalization states audited. No GPU, natural language model, arbitrary pretrained Transformer weight transfer, or theoretical function-class impossibility claim.

Files include [Japanese frozen plan](EXPERIMENT_PLAN_JA.md), [original protocol](PROTOCOL.json), [stage B/C/D protocols](DIRECT_PROTOCOL.json), [reported results](RESULTS_SUMMARY_JA.md), and [summary table](results/RESULTS_SUMMARY.csv). Training/replay source and tests live in this folder. **All raw run CSVs, inference model payloads and logs are in the downloadable conversation evidence ZIP**, not necessarily all on GitHub.

Full-evidence ZIP SHA256 `b5da134cf6ce7c34f2484d89eba88e492255116b76724359d25299ac0ec624a9` (197 entries, 5,523,653 B).
Source ZIP SHA256 `57185fa7589b6c757d5a508a57e859ed31880413a9e513fd827ca0704c8db3f6` (50 entries, 108,911 B).

**No changes to main, canonical MA registry, previous DMFFN-001 positive synthetic result, or any historical MA status.**
