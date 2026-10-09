# Source-informed nonlinear FFN Mirror tests — FF-NL-004/005/006

Date: 2026-10-09 JST  
Scientific status: **3/3 FAIL_DEVELOPMENT_GATE**. This is a separate research intake, **not a formal MA status**. Existing live MA registry and main branch are unchanged.

## Motivation and fair question

Earlier rotational and FFN Mirror candidates could not beat strong native controls. This program gives both Mirror and native rank-4 LoRA the **same twelve independently learned source-domain internal FFN differences**, then trains task-specific codes from **192 target training examples** on eight withheld digit image-corruption conditions. Source adapters are trained on source tasks only, never fitted to target audit checkpoints. The shared-basis models additionally require **240 source basis-fitting updates** beyond the common 12×210 source-adapter updates; the extra compute is not free.

Original base network: 64→96→64→10 frozen clean-digit classifier, with trainable/adapted internal 64×96 FFN weight. Data: packaged sklearn 8×8 digits with synthetic rotation, blur, occlusion, shift etc. CPU PyTorch 2.10.0, FP32, one thread. Two development seeds per experiment; fresh worlds stayed sealed.

## Registered experiments and measured findings

| Trial | Tested candidate | Candidate mean CE (nat/example) / accuracy | Nearest strong comparator | Verdict |
|---|---|---|---|---|
| FF-NL-004 (6601/6602) | **nonlinear 8D code**, including four noncommuting quadratic interactions | **2.0459 / 59.51%** | direct linear 8D code 2.0293 / 59.35%; source-warm LoRA4 **0.5818 / 81.49%** | FAIL |
| FF-NL-005 (6701/6702) | **shared rank10 core + private rank1** | **0.5740 / 81.34%** | shared core only 0.6741 / 78.34%; source-warm LoRA4 **0.5301 / 82.36%** | FAIL |
| FF-NL-006 (6801/6802) | same source-information family at **540 rather than 180 updates** | **0.4774 / 84.96%** | source-warm LoRA4 **0.4660 / 85.46%**; shared rank10 0.5466 / 82.22% | FAIL by frozen all-seed quality rule |

All averages are across **two development seeds × eight target domains**, not confidence intervals. FF-NL-006 also tested private rank2: **0.4616 CE / 85.67%**, but eight-task storage of 82,950 bytes is higher than warm LoRA4's 80,436 bytes. It therefore does not supply a free quality improvement.

All actual NPZ inference bytes include the shared frozen model, source-derived basis/means, full task-specific codes/factors and metadata:

| Method (FF-NL-006 at 540 updates) | 4-task actual NPZ | 8-task actual NPZ |
|---|---:|---:|
| source-warm LoRA rank4 | 67,460 B | 80,436 B |
| shared core rank10 | 65,370 B | 68,746 B |
| shared core rank10 + private rank1 | 69,926 B | 77,830 B |
| shared core rank10 + private rank2 | 72,486 B | 82,950 B |

FF-NL-006 primary rank1 was narrowly below a **per-seed** preregistered accuracy margin in seed6801 (difference vs warm LoRA −0.01044566, threshold ≥−0.01). Seed6802 passed; overall verdict remains FAIL. Fixed-update gains are evidence of learning dynamics, **not a proof of asymptotic capacity**.

## Separate POST-HOC mechanism audit (not result promotion)

1. **Physical storage crossover:** without additional training, repacked FF-NL-006's eight stored target codes into real K=1..8 NPZ inference files. At **K=6**, rank1 shared/private is **73,878 B vs 73,948 B** source-warm LoRA; K4 is *larger* and K8 saves **3.24%**. All 16 regenerated K4/K8 files match the stored experiment files in **exact byte length and SHA256**. This is a storage-only break-even, not comparable accuracy evidence.
2. **Noncommutative interaction ablation:** holding trained FF-NL-004 m codes fixed, turn only the nonlinear coefficient beta off and replay the *same development samples*. CE increase from removing the nonlinear term was on average **−0.005433 nat/example** (negative means the nonlinear term hurts); positive in 9/16 tasks, negative in 7/16. Source seed6601 mean +0.01157, seed6602 mean −0.02244. The quadratic contribution was **5.73%** of the norm of a task core delta.
3. Four commutator matrices raise the **linear span** of directions from 8 to 12 in both seeds, but the m→core differential is a 100×8 Jacobian with rank at most eight; do not call this 12 independently tunable local functions.

None of these post-hoc probes changes the frozen scientific status.

## Verification and evidence

- Three studies: **368 development metric rows, 92 serialized inference banks**; replayed with zero accuracy difference and max CE discrepancy **9.54e−7**.
- **26/26 tests passed**: 16 + 4 + 3 for FF-NL-004/005/006, 3 post-hoc storage/ablation tests.
- Development source code, frozen protocol, and result CSV SHA256 verified identical to the original GATE records. No fresh test/audit seeds opened.
- CPU: AMD EPYC 9V74, quota4 CPU, no GPU, clock not locked, PyTorch FP32 eager, torch1 thread. Folded batch1/64 throughput was measured but no GPU or LLM throughput claimed.

**Conversation evidence artifacts**, not GitHub-hosted model files:

- `FFNL_2026-10-09_FULL_EVIDENCE.zip` 15.5 MB; SHA256 `9f4f4e21b2c0467fd980b01d09150699bc54ce813eaf8b7c73c94760f60cfe81`.
- `FFNL_2026-10-09_SOURCE_REPORTS.zip`; SHA256 `a26d2bfef1579fb62f052d2a87e21b6c701a4f33ec31e5308760db4c727bfd7a`.
- `FFNL_2026-10-09_REPORT_JA.md`; SHA256 `b2d9b3a6cb7033c4b70e459cb92393f18a767ad84d4f6b6ea946d0cc8fc0d5d6`.
- `FFNL_2026-10-09_SUMMARY.csv`; SHA256 `ea1be1025289ce8648fac53989dc065b3b4a15dc348a22645505c84d4814149f`.

**Repository boundary:** This branch deliberately records the conclusions and immutable evidence hashes. It does not contain all executable Python files or the 92 NPZ payloads; obtain the conversation ZIP for the full source and replay command. Do not claim that the GitHub file alone is a runnable reproduction.

## Next experimental decision

Do not treat method count or a higher linear span as independent capacity. Before further narrow m geometries, require a new preregistered *cross-family* target benchmark where both Mirror and native source-warm LoRA receive the same source-task pretraining and deployment accounting. Include a function-space utility target, hold out **entire task families**, add shared/private fallbacks, report real bytes and measured active compute, and preserve all null results.

Existing line: `research/function-first-nonlinear-ffn-20261009`. Current independent evidence branch: `research/ffnl004-006-strong-control-20261009`. **No main merge.**
