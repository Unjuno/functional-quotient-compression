# MA-1176 — ExpertFlow / DyMoE / MoE-Infinity direct serving controls

**ADDITIONAL CONTROL ONLY.** This is not a new MA and is not a change to an earlier frozen protocol or scientific status. Current worker MA-255 and its original queue remain untouched. These stronger native controls should be introduced only through a **future preregistered protocol amendment** if that MA is activated.

## H — falsifiable extension
A Mirror-encoded physical Expert group may reduce actual persistent expert bytes and amortize CPU→GPU DMA over multiple logical m roles, but the benefit must survive a native predictive expert router/cache/scheduler and mixed-precision prefetch implementation on the SAME hardware.

## T — standalone test recipe
Freeze a specific MoE checkpoint/router, tokenizer, K physical groups, source-only query distribution, two GPU HBM budgets and PCIe/NVLink topology. Compare (a) MoE-Infinity activation-aware cache with host-memory + SSD resident-set policy, (b) ExpertFlow precomputed path predictor+token scheduler+cache, (c) DyMoE importance/depth mixed-precision and lookahead, (d) PA434/435/436 where source runnable, (e) these same predictors and cache discipline applied to Mirror physical blocks with m roles, (f) native ordinary factorized logical roles with equal bytes. Record predictor false positives and miss penalties, PCIe/SSD bytes, pinned host memory, CUDA stream DMA overlap, group swaps, TTFT/TPOT, exact task quality and VRAM high watermark, invalid downstream KV paths. Split development route traces from fresh route traces by independent prompts, not token truncations. Only call a strategy better if REAL CUDA stream-synchronized end-to-end latency/quality beats strongest deployable native at equal HBM bytes.

**External native sources (verify exact algorithm before claiming paper reproduction):** https://doi.org/10.1145/3770743.3804292; https://arxiv.org/abs/2603.19172; https://github.com/EfficientMoE/MoE-Infinity. Additional PA refs: PA450; PA451; PA434; PA435; PA436; PA439.

**Source/dev/fresh:** dev seeds 11/12/13; completely disjoint fresh worlds 101/102/103/104/105 ONLY if the original MA's audit is unopened; otherwise create entirely new task seeds/identities and a separate frozen amendment. Never alter a prior trial retroactively. All source adapter/selector calibration, thresholds and weights come from permitted source+dev data. Report all five fresh worlds, no audit stopping.

**Account every paid state:** copied cache, adapter library, common decoder, readout code, CPU-pinned staging, GPU high-watermark, metadata and resident folded copies. Record optimizer updates, MACs and wall P95. Compare actual output quality, not just weight reconstruction and not solely byte count.

## D — PASS / FAIL / UNCERTAIN
**PASS:** marginal m effect beats the strongest listed native alternative at native-equivalent task quality and strictly improves actual bytes or runtime with no >10% P95 regression across >=4/5 disjoint fresh worlds; strongest same-byte simple code must not explain all benefit.
**FAIL:** FAIL if native ExpertFlow/DyMoE/MoE-Infinity at matched VRAM has equal or better TTFT/TPOT/quality, or Mirror queues speculative transfers but requires costly unfused per-role compute.
**UNCERTAIN:** absent native implementation, uncertain task labels/seed firewall, missing real serializer bytes, no real GPU offload when a GPU claim is made, or interval spanning the declared margin. Do not promote prior statuses.

## C — alternative explanation
Pre-gated native MoE already hides transfers and skew-aware mixed precision reduces bytes; Mirror m may impose extra transform/alias overhead without reducing actual transferred HBM state. A static simulator cannot prove real overlap.

## U — uncertainty and units
Model outputs and trained weights are normalized and dimensionless; m is dimensionless; quality may be nat/token or dimensionless AUROC, as defined per task. Physical storage S is nonnegative byte integer (practical unit), wall latency t in SI seconds, trained-step count as integer. Do not combine these into one opaque number. Use paired independent task/world CI and numerical replay checks; `u_c^2=u_seed^2+u_task^2+u_num^2` only under independence; add covariance terms otherwise. k_cov=2 gives an indicative expanded interval, not a 95% guarantee at n=5.

**Measurements:** TTFT ms, TPOT ms, resident HBM byte, physical PCIe/SSD transferred byte, copy count, prefetch hit/precision/recall, output NLL, CPU-GPU latency. Exact original native methods must be verified from released source prior to publication-grade claims.
