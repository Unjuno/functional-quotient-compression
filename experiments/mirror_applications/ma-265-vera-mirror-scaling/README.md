# MA-265 — VeRA shared scaling codes vs Mirror scaling views

Status: **FAIL (fresh actual-payload gate)**. Evidence lane: MECHANISM / STORAGE. Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`.

## H / Mirror insertion

H: If task adapters lie on a low-dimensional orbit of VeRA's shared scaling vector, a shared base scale plus per-task Mirror angles can retain adapter quality with fewer bytes than independent VeRA scaling vectors; unrelated codes should need VeRA/private state.

> **Mirror insertion:** this experiment adds a task angle `m_t` to the scale vector over one frozen shared VeRA basis so that orbit-related adapters are expressed without storing a full scale vector for each task.

Native method: VeRA shares frozen low-rank matrices and learns scaling vectors. Mirror changes the task scaling code through a 2D orthogonal rotation in code space. Physical objects: frozen shared basis matrices A/B. Logical objects: eight task-specific low-rank regression adapters.

## Prior-art delta

PA18 is mandatory control. This experiment tests compression of VeRA's per-task scale vectors under an explicit functional orbit; it does not claim shared random bases or vector scaling as Mirror inventions. Controls include native VeRA code vectors, shared hard-tied adapter, per-task LoRA factors, and independent dense task maps.

## T

Synthetic 16D-to-12D regression, rank 4 frozen Gaussian A/B, eight tasks. Aligned condition: task scale codes are rotations of a common code in the first two coordinates. Unaligned condition: independent random scale vectors. Fit codes after least-squares task-map recovery from 512 examples/task; evaluate 1024 held-out examples/task. Development seeds 19,37; fresh seeds 109,229,313,421 locked in protocol. Post-fit screen, zero optimizer updates.

## Gates

PASS: at least 3/4 fresh aligned worlds, Mirror MSE <=1.10x VeRA, payload <=80% of VeRA, and below hard tying; unaligned condition documents boundary. FAIL: Mirror aligned MSE >1.25x VeRA on >=3/4 seeds or byte reduction <10% at matched quality.

## Accounting

Actual uncompressed NPZ bytes include A/B frozen bases, shared base scale, per-task codes/angles, LoRA factors, and dense maps. Mirror base scale and task angles are stored in one packed inference coordinate array (frozen before fresh evaluation) to avoid per-array header overhead. Inference MAC proxy includes basis projection plus low-rank reconstruction. View decode latency is reported separately and not optimized.

## Results — H / T / D / C / U

FACT: Across fresh seeds 109, 229, 313, 421, aligned-orbit mean MSE was numerical zero for VeRA and Mirror. VeRA payload was 3,654 bytes; Mirror was 3,494 bytes, a 160-byte / 4.4% saving, missing the preregistered 20% reduction gate. Hard-tied MSE averaged 0.153 at 3,430 bytes. Rank-4 per-task LoRA matched numerical-zero MSE at 9,434 bytes; independent dense task matrices were 12,546 bytes. On independent scale codes, VeRA remained numerical-zero at 3,654 bytes while Mirror MSE was 0.924 at 3,494 bytes. Each seed/condition fit used 4,096 examples and zero optimizer updates. The packed Mirror payload roundtripped exactly.

INTERPRETATION: Mirror reduces the task-address state from 32 VeRA code scalars to 12 scalars, but the frozen basis and shared map dominate total storage, so overall savings are only 4.4%. Independent task code behavior is lost.

HYPOTHESIS: Functional-code compression can be meaningful only when code bytes are a substantial share of total inference payload, or when saved code bytes enable more task adapters at a fixed budget.

COUNTER-HYPOTHESIS: Better vector quantization, a smaller shared basis, or a nonlinear learned code manifold could change the frontier; this experiment tested only one angle chart.

UNCONFIRMED: trained PEFT adaptation quality, real datasets/LMs, fixed-byte scaling of task count, quantization interactions and decode throughput.

Decision: FAIL the predeclared storage gate; preserve the aligned code-compression and unaligned failure evidence.

## Protocol variant reconciliation

The separate `protocol_variants/fixed_update_failure/` directory retains another frozen MA-265 screen. That 1,000-update development-only protocol also failed: native VeRA mean MSE was about 0.117 versus Mirror 0.232, while Mirror saved only about 8% payload; fresh worlds 26502–26504 stayed sealed.

The two protocols test different teacher/code families (post-fit rotation of VeRA scales versus trained diagonal rank coefficients) and are not pooled as replication data. Both bound Mirror's current case: a matched rotational orbit can reduce code bytes, but the overall payload saving missed the predeclared 20% gate, and native VeRA remained stronger on the distinct diagonal-scale training task.

Source branches: fresh post-fit `research/ma-265-vera-mirror-scaling-20261008` (verified result commit `9e3442255afbd309d29100c32f006df216b2102b`); development-only failure `research/ma-265-vera-mirror-scaling-replication-20261008` (report branch `b2b59632beefef0d7713e16367364214b3ff0c04`).
