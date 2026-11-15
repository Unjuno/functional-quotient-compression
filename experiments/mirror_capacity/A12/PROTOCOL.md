# A12: disentangling learned scaffold and unseen-task code identification

Frozen design date 2026-11-15, before A12 training/audit. Status: EXPLORATORY (A11 led to this question).

## Research question
Why did A11 tent/sine teacher tasks fail the joint NRMSE≤0.20 quality gate: insufficient shared-scaffold meta-learning, insufficient 32 labeled samples for each new code, insufficient code optimization, or architectural capacity/other optimization limits?

## Primary causal interventions (one axis at a time; paired on same world/seed/model)
- Teacher families: `tent`, `sine` exactly from A11; no Mirror used to generate targets.
- Fresh worlds: 1200,1201,1202. Independent seeds: 0 and 1.
- Methods: shear Mirror, direct Shift, FiLM (unchanged A11 `Student` and 1024B cap, all learned tensors counted).
- Meta optimization: checkpoints after **600 and 2400 uninterrupted AdamW updates** from the identical random initialization and minibatch streams. Learning rate 0.01, weight decay 0, batch 128, clip 5.
- Support data: **32 vs 256 nested labeled examples per new task**. First 32 coordinates and labels must be bitwise equal to A11 support generator. Teacher parameters never enter the student initialization or gradients except to produce training labels.
- Adaptation compute: checkpoint after **200 vs 800 uninterrupted AdamW updates** of the new-code parameters, starting from the identical initialized codes, matched task/batch RNG streams per support size. New task codes are learnable; all shared and old-task weights frozen, verified bitwise before/after.
- Total = 2 families × 3 worlds × 2 seeds × 3 methods × 2 meta checkpoints × 2 support counts × 2 adaptation checkpoints = **288 final model files**. Meta checkpoint is also serialized to allow exact stage audits; 72 meta-stage files. Full/partial results remain separated.
- Seen-task meta training: 1024 input coordinates, 16 tasks; old-task dev 256; new unseen task IID 256 and OOD 256. Same teacher-family for seen/new IDs (no unseen-family test).

## Measurement and decision gates
- Quality gate: all of seen-dev, unseen-IID and unseen-OOD NRMSE ≤0.20, computed per world as mean over the two optimization seeds. All values are measured after all final model checkpoint bytes/hashes are locked; final tests do NOT tune configurations.
- Paired descriptive effects: `new_IID_NRMSE(600,256,800) - new_IID_NRMSE(2400,256,800)` for meta; `new_IID_NRMSE(2400,32,800) - new_IID_NRMSE(2400,256,800)` for supports; `new_IID_NRMSE(2400,256,200) - new_IID_NRMSE(2400,256,800)` for adaptation updates. Positive difference means intervention improves.
- Support-sample bottleneck H_S: support increase produces ≥0.05 absolute or ≥25% relative IID gain in ≥2/3 world means for a method/family, without worsening seen dev (fixed) and OOD by >0.02. Otherwise not supported under tested setup.
- Scaffold/meta-training bottleneck H_M: longer meta improves seen-dev by ≥0.05 absolute AND unseen-IID by ≥0.05 absolute or ≥25% relative in ≥2/3 worlds; all at fixed support256/adaptation800. Otherwise not supported.
- Code optimizer limitation H_C: 200→800 adaptation updates improves unseen IID by ≥0.05 absolute or ≥25% relative in ≥2/3 worlds with fixed meta2400/support256. Otherwise not supported.
- If multiple mechanisms meet these criteria -> mixed. If none -> under tested capacity and training design, *unresolved structural/optimizer limitation*; cannot claim mathematical impossibility. With 3 worlds, do not claim inferential population intervals.

## Additional positive-control diagnostic, outside model comparison
For each new teacher function, separately fit an explicit tent or sine formula from exactly the 32/256 support input-output pairs using a deterministic multistart least-squares or frequency scan, without reading true teacher coefficients. Evaluate its IID/OOD NRMSE and record its parameter bytes separately. This family-informed oracle is **not a fair model competitor**: it is evidence about whether 32 samples could in principle determine this known function family. It cannot show generic few-shot identifiability.

## Evidence/audits
- PyTorch 2.10.0+cpu, FP32, 1 numerical thread/worker; CPU model and observed non-locked clock logged at run. Workload runtime descriptive only; step-count matching is not FLOP matching.
- Files: 32-byte header + all learned FP32 parameters + SHA256 32-byte trailer, same A11 serializer.
- Tests written before implementation: nested support prefix and no oracle-coefficient leakage, adapted frozen weights unchanged, 600-step compatibility with A11, lock/hash/roundtrip check before any audit, no use of dev/IID/OOD labels during training, reproducibility.
- No audit-based selection, reruns after seeing outcomes, or merging smoke with main counts.
- Provenance for inherited A11 implementation is `experiments/mirror_capacity/A11/a11.py` on main, and A11's separate evidence ZIP. Independent A12 source includes only its delta; repository assumes A11 remains present.
