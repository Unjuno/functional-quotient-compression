# Antipodal Mirror View distance, multi-reflection state banks, and functional value
Date: 2026-10-10. **Research intake, not a completed registered MA ID.** Main and official MA status ledger unchanged.

## Research question
Does making views maximally separated, then combining multiple reflections, efficiently create useful, distinct functions from a shared physical basis? Distinguish (1) representation distance, (2) useful task quality and diversity, and (3) actual byte/runtime savings over native shared-linear/signed controls.

### A. Learned K=8 real-image tasks, rank6
Sklearn 8x8 digits, three development split seeds 8301–8303; same-input eight binary tasks. Paired world consists of 4 predicates plus their complements; unpaired world contains eight non-complement digit predicates. Shared 64→48→32 ReLU trunk fitted on training images, then frozen; all role models trained 320 updates on same examples/minibatch seeds.

| Method | paired BCE nat/label | unpaired BCE |
|---|---:|---:|
| Independent eight heads | 0.19149 | 0.17942 |
| Ordinary linear low-rank r6 | 0.17056 | 0.16901 |
| Householder mirror r6 | 0.17464 | 0.16845 |
| Projectively spread Householder | 0.17489 | 0.16640 |
| Ordinary output-spread Householder | 0.17427 | 0.17038 |

Spread increased projective normal separation from 0.7195 to 0.8932 (paired) and from 0.7499 to 0.9183 (unpaired), without a stable >=0.005 BCE benefit. Rank6 basis/code metadata also made these K8 banks larger than independent heads. **H1 distance-effect FAIL, H2 mirror-specific quality/byte/latency gate FAIL.**

### B. Independent rank4 K=4/8/16/32 scaling
New seed splits 8501–8502. Predicates are disjoint as task IDs but share one 10-class digit source. K32 unpaired: reflection 0.3587 BCE/24,626B vs independent 0.2159 BCE/26,570B. K32 paired: reflection 0.3491 BCE/24,630B vs independent 0.2194 BCE/26,577B. Byte reduction ~7.3% fails quality. Native known-pair signed heads achieve paired 0.2194 BCE at 24,612B: smaller and more accurate than reflection. **Both preregistered K-scaling gates FAIL.** This is not 32 independent semantic data sources.

### C. Multiple independent sign flips: exponential *distinct* logical states
Synthetic Gaussian x dimension32, shared rank r=3/4/5, all K=2^r sign strings, 3 development seeds 8701–8703, 1024 training/2048 clean-output test per seed. Fit B in W=B S^T/sqrt(r). All sign addresses packed into actual self-describing inference NPZ. Aligned rank5/K32: signed bank **2,025B and MSE 0.0000131**, independent heads **4,847B and MSE 0.0000804**: ~58.2% storage reduction while maintaining quality. Independent random teacher K32: signed bank MSE **0.8284**, independent heads **0.0000819**; real rank5 code ~0.5561. This stark boundary is expected when the teacher's true function matrix rank is higher than the representation rank.

**Critical exact alias:** 18/18 NPZ file comparisons between the proposed Mirror sign bank and a conventional native signed-basis implementation are **bit-for-bit identical**, including SHA256 and decoded outputs. Therefore **H3 aligned feasibility PASS; H4 unrelated-independent transfer FAIL; Mirror-specific superiority FAIL**. A single involution has at most two orbit states. With r independent sign axes, the 2^r state vectors can create 2^r *distinct* linear functions yet their span has dimension at most r. Neither number of codes nor maximal weight-space distance establishes 2^r independent trainable skills.

### Mathematically decisive gauge counterexample
For an odd activation tanh, flipping both first and second layer weight tensors changes their Euclidean distance by twice the full weight norm while leaving network outputs exactly equal. In this numerical check the inversion distance was 23.1819 while the maximum output error was 0. Householder normals u and -u produce exactly the same reflection, so raw code separation can count functional duplicates.

### Evidence / reproducibility
- CPU: **Intel Xeon Platinum 8573C**, clock snapshot **2299.998 MHz, not locked**, cgroup 4 cores / 4GiB, PyTorch 2.10.0+cpu, Python 3.13.5, NumPy2.3.5, sklearn1.8.0, single torch thread, FP32 learned heads (synthetic regression solved float64, stored FP32).
- 3 *pre-locked* protocols; source hashes were pinned before development outcomes.
- 17/17 unit tests pass; **211 saved inference NPZ files** with size/SHA exact replay and **1,101 numeric results recomputed with maximum error 0**. 18 file-level native sign aliases exact. This is numerical replay, not 211 retrainings.
- Unopened fresh IDs: 8401–8403, 8511–8512, 8721–8722. Both natural-data experiment gates fail at development. Do not promote synthetic aligned evidence to natural-language or GPU inference.
- Primary numerical archive **MIRROR_ANTIPODAL_2026-10-10_FULL_EVIDENCE.zip** (3,307,980 bytes, SHA-256 `3aa352ff12abfb040708cdcbc4f043cc9a07ab2c5c641dcf97628d1d52115bae`), plus lightweight source/reports **MIRROR_ANTIPODAL_2026-10-10_SOURCE_RESULTS.zip** (102,861 bytes, SHA-256 `8c6775069a83e5ded5fdb625bc02e445a65e4cb46bdb4a0584a3b958c83961aa`). Both are **conversation artifacts, not GitHub-hosted archives**; this branch stores the summary/selected aggregate data/hash index only. Full experiment.py, experiment_scale.py, experiment_orbit.py, tests, protocols, NPZ, and logs are inside the archive.
- Prior-art direct control: Yuan, Liu, Xu, *Householder Reflection Adaptation*, NeurIPS 2024, DOI 10.52202/079017-3606. Householder and signed/orthogonal coordinate families are not Mirror-native inventions.

### Next decisive experiment
Train a diverse bank of *naturally acquired* source-only nonlinear task functions. Estimate whether task differences admit a low-entropy sign/Householder decomposition while preserving task logits and OOD behavior. Compare against **native learned signed bases, unrestricted shared low-rank codes, source-warm LoRA, HRA and explicit private fallback** at the same training-source budget, full serialized inference bytes and compute. Only a fair Pareto win beyond exact native aliases should change a Mirror-specific status.
