# MA-1174 — 事後フォレンジック（2026-10-21）

> **Evidence classification:** Independent after-the-fact code / checkpoint / split audit. **Not** a new preregistered positive result. Historical result remains mechanism PASS, prospective learning quality FAIL 0/6, Mirror-specific quality FAIL 1/6.

## Original experiment clarified
- This was **8-dimensional Gaussian synthetic function regression**, not a language model. Two separate synthetic teacher worlds, not simultaneous multi-task learning.
- Fixed four nodes, two Mirror views, 32 directed 8→8→8 logical MLP edges, two recurrent passes; 128 shared physical coefficients plus 128 learned addresses, 32 gates and 32 biases = 320 independently learned scalars.
- Only **hard edge removal** was implemented (32→24, at updates 90 and 170, cutting four each time). Node count, view count, Loop count, output construction, edge regrowth and rerouting were **not** optimized.
- Original candidates were (i) sign fluctuations of each gate derivative between two minibatches, (ii) lowest gate, (iii) lowest absolute gate derivative, and (iv) random. The first **is not actual edge-to-edge conflict of shared-core parameter gradients**. “Taylor” means raw gate derivative, not exact hard-off loss influence.
- Five models (keep + four cuts) were cloned from the same AdamW state and trained **six further updates**, using identical minibatches. `mirror_lookahead` picked the lowest-dev-loss **cut**, even when the no-cut reference was better. `mirror_guarded` could reject all cuts.

## Forensic findings from original logs
1. **No 6-step beneficial cut:** 12/12 interventions had the best cutting proposal scoring **worse than keep** on dev. Best-cut minus keep: +0.00091 to +0.02184 normalized MSE, mean +0.00635. Nevertheless `mirror_lookahead` forced all 12 cuts. `mirror_guarded` rejected 12/12.
2. **Short horizon misses later adaptation:** exact reproduction from the same model and optimizer state extended the four cut candidates (and keep) to 6,20,50,90 future updates using **development only**. The *hindsight* best cut beat keep at 0/12, 1/12, 4/12, 6/12 respectively. The originally selected **six-step** candidate actually beat keep at 0/12, 0/12, 2/12, 2/12. Only 3/12 six-step candidate winners remained the best cutting candidate after 90 updates. At the first cut event, horizon170 hindsight found beneficial cuts in 2/6 cases, whereas the six-step selected cut beat keep in 0/6. **These are exploratory hindsight observations, not deployable selection performance.** Original six-step numerical loss and end-of-training dev loss replays matched to 1e-6.
3. **True edge-shared-core VJP differs from the proxy:** for 18 trained checkpoints (keep/lookahead/low_gate × six world-seed pairs), separately differentiating each edge's contribution to the 128-coefficient shared core reproduced the whole core gradient with relative discrepancy around 1e-7. Negative pairwise shared-core gradient cosine prevalence increased in **5/6 final trained lookahead models** versus corresponding all-keep models, not decreased. This cross-trajectory comparison is **not causal evidence that cutting increases harmful interference**; negative cosine is not automatically harmful.
4. **Virtual function diversity remained low:** after lookahead, 24 logical generated 128-value edge-weight vectors had **median spectral-entropy effective rank 2.46**, and **median 99.84% of their squared singular-value energy lay in the top four directions**. Numerical rank at relative 1e-3 was 11–16. Spectral rank of weights is not the functional Jacobian rank; still, large virtual multiplicity did not manifest as correspondingly diverse edge-weight directions.
5. **Aligned teacher's eight edges were not identified:** the learned 24-edge masks retained respectively 6/7/5 of eight exact teacher edges, leaving 18/17/19 non-teacher edges. Masks need not be functionally identifiable due symmetries, so this is an addressing diagnostic only.
6. **Strong controls dominate:** Fresh NMSE median mirror-lookahead .13491 vs 314-parameter conventional MLP .02145; CPU B1 273.99µs vs 14.54µs (1 thread, FP32, unlocked clock). A *posthoc* 72-parameter least-squares linear control beat mirror-lookahead in **all three independent-MLP teacher worlds**, though not in the three aligned worlds; this control was absent from the preregistration.
7. **Retrospective from-start control uses a hindsight-selected mask:** the 24-edge mask is copied from the already completed lookahead training; not an online algorithm that knows a mask in advance. It beats lookahead Fresh 6/6 but cannot justify claiming that all structure evolution is worse.
8. **Compute matching was approximate:** 260 actual + 60 cloned probe updates for lookahead vs 320 actual updates for keep_extra are *not* identical MAC budgets (virtual-MLP proxy ~145.62M vs ~167.77M, before weight decoding and backward costs). A fair Pareto comparison needs FLOPs-to-target and wall-time matched runs.
9. **Oracle replay is a tautological software check:** it copies the same-family teacher's entire parameter state and mask into an identical student implementation; this does not prove arbitrary-MLP representation or architecture discovery.

## Primary scientific verdict
**Do not promote the original experiment.** It shows gradient direction and active compute change after hard cuts. It does **not** demonstrate better rule learning, Mirror-specific quality benefit, effective structure discovery, nor generalization to a causal LM.

## Next prospective experiment requirements
- Construct true shared-core simultaneous multi-rule workloads and controls designed to exercise *competing* gradients, with independent/non-Mirror teacher controls.
- Rank interventions using genuine per-edge shared-core gradient VJPs, and include **KEEP as an explicit decision** at every step.
- Preregister longer H probes or a separate learned proxy for **long-horizon future quality**, and evaluate on new sealed fresh worlds (these posthoc development observations cannot become fresh).
- Enforce equal inference packet bytes, actual training FLOPs and wall-clock, include same final topology and ordinary dense FFN controls.
- Measure function-level diversity / causal specializations; actual node/view/Loop search requires its own subsequent experiment.
- Preserve all earlier failures and do not merge this experiment into `main`.

## Verification and uncertainty
Original PyTorch 2.10.0+cpu / Python 3.13.5 source checksum and frozen protocol SHA-256 `80308193fd6b22aabe84f6e27c9f8350dafd792b433d35b01f7cff357d5207e8` verified. Original 7/7 tests pass; all 66 checkpoint hashes match; original CSV checksum matches; dev-only counterfactual 330 records are replayed, and all 18 edgewise gradient VJP sums match within about 2e-7 relative.

Six paired world-seed trials are small and heterogeneous; descriptive lookahead minus same-final-mask Fresh NMSE mean +0.005836, standard-error-like u_c≈0.000999 and k=2 descriptive interval +0.00384..+0.00783, **not** a calibrated population confidence interval.

Supporting local reproducibility archive in this conversation: `MA1174_posthoc_audit_bundle_20261021.zip` with `posthoc/horizon_audit.py`, `gradient_audit.py`, `secondary_audit.py`, result CSVs, and full Japanese postmortem.
