# MLP Mirror — usage decision and conditional adoption gates (2026-10-17)

**Research recommendation: keep Dual-Input Mirror SwiGLU as an OPTIONAL non-linear role adapter, NOT a generic MLP/Transformer replacement.** This document audits four experimental families including the new readout-first MLP-USE-002; previous MA statuses and main remain unchanged.

## 1. What actually worked

- **DMFFN-001 (Givens-aligned synthetic teacher):** 3 gate Views × 3 up Views produce 9 logical combinations with one shared gate/up/down SwiGLU. At 648 parameters / 5,081 actual bytes, dual Mirror12 beat equal-parameter native gain12 on 5/5 independent synthetic worlds (MSE 0.0485 vs 0.1878 world median). Unseen gate×up pairs were predicted more accurately. Unrelated independent functions failed, and independent 9-FFN control remained better even with extended training. This establishes compositional structural fit, not 9 independent experts.
- **DMFFN-003 (natural handwritten-digit independently learned FFNs):** sharing known tasks can achieve good classification at fewer bytes; on **unseen natural task combinations** Mirror failed badly versus a very simple shared model. More training did not solve generalization. Thus a Givens-aligned teacher is a strong optimistic bias.
- **MLP-ROLE-001 (application test):** Four roles, fresh 168 models in 3 disjoint random splits × 2 seeds × 7 methods, later 144 models on 3 new splits. Related digit-attribute classification: Mirror BCE 0.11308 vs same-count native gain 0.13772, better in all three paired worlds; BUT a native shared SwiGLU with task-specific down/readout BCE 0.10321 at 15,573 B is both **better and smaller** than the Mirror 15,885 B. Image denoising Mirror MSE 0.03676 versus gain 0.03570 and native readout 0.03292. Diabetes four endpoints derive from ONE true response, and forecasting is a synthetic nonlinear autoregression, not a set of independent real tasks: their Mirror-specific gains were negligible.
- **MLP-HYBRID-001 (native private fallback):** On fresh digit splits, 220 models and 60 new-world long-budget models. Attributes: Mirror+rank2 selected-checkpoint BCE 0.07680 vs equal-parameter gain+rank2 0.08867, but frozen joint quality/worst-view gate **FAIL**. Mirror+full 0.06904 at 16,773 B, versus pair-readout 0.06884 at 15,573 B. Denoising: Mirror+full MSE 0.03315 at 24,133 B vs gain+full 0.03283 at same bytes and pair-readout 0.03286 at only 21,589 B. Strong native baselines remain competitive or dominate.
- **MLP-USE-002 (new readout-first test):** Freeze one shared gate/up parent and FOUR private readout matrices, then add 32 learned Givens/ gain/ pair-specific Givens parameters; 48 runs on 3 NEW digit splits, 700 AdamW updates. For attributes, separate medians 0.05336 (readout+Mirror), 0.06248 (readout+gain), 0.06633 (readout-only), 0.05741 (readout+pair Mirror); BUT paired-world stricter win gate was only 1/3 vs each competitor: **FAIL**. Denoising strict paired gate 0/3 for all. Native readout forward 0.0416 ms vs Mirror 0.1600 ms on eager CPU batch64.

**IMPORTANT:** Compare a method against its control using *ratios within the same world/seed design*, NOT the ratio of independently aggregated medians. Even when Mirror's separate world median appears lower, a preregistered >=2/3-world joint improvement may be false.

## 2. How should Mirror actually be used?

Choose native readout or bias first. Consider Mirror only when **(a)** several logical functions use the SAME input features and expensive gate/up matrices, **(b)** task-specific readouts cannot explain the differences, **(c)** the role variation lies near a compact nonlinear gate×up transform family, and **(d)** at matched inference byte + training FLOPs it beats native gains, LoRA, task-specific heads and a private fallback on NEW tasks.

Recommended selectable family (the budget controller must measure all alternatives):
1. shared FFN + task-specific output readout (native default);
2. + small task bias;
3. + native gain/FiLM, native rank1/2 corrections;
4. + factorized Dual-Input Mirror (only if non-linear residual remains);
5. + selectively private FFN channels or full private block for out-of-orbit roles.

An automatic controller should choose improvements by **held-in development validation-loss reduction per incremental *actual saved byte*** with independent constraints on incremental active FLOPs and training FLOPs, not simply increase Mirror angle count. Previous automatic angle growth failed.

## 3. Application priorities

**P0: natural independently trained FFN checkpoint compatibility.** Align SwiGLU hidden channels only by valid common permutations; compare heldout FUNCTION output MSE (not raw parameter distance), posthoc orbit fit and few-shot code learning separately. Distill pretrained/independently learned natural FFN roles; do not fit parents using sealed test roles.

**P0: paired, non-linear residual benchmark.** Start from strongest shared features + task-specific output readouts; freeze parent as necessary; show a new dataset where gate×up inter-role interaction is not expressible by the head. Compare exactly equal-byte Mirror, native gain, low-rank and task-private additions over a fixed learning-compute grid. The current readout-first pilot failed; larger/different *predeclared* tasks are required.

**P1: sparse expert MoE.** Use top-1 / top-2 routing of factorized gate/up pairs; pay router and all active costs. Cartesian K_g*K_u is the number of addresses, not independent model capacity, and multiple pair computations are necessary to amortize shared projections.

**P1: MTP draft and looped Transformer.** Insert Mirror only after a shared final attention/cache boundary initially. Evaluate real target p/q acceptance, rejection correction, KV copy/rollback, and true decoding tokens/s. In a recurrent block, every loop takes an updated input: gate/up projections cannot simply be reused from a previous loop.

**P2: dynamic capacity controller.** After the native frontier is established, run a held-in validation controller over {readout, gain, Givens, rank2, private} and only retain an option that improves generalization per added byte and FLOP, with rank/angle allocation stabilized across seeds.

## 4. Scientific boundaries and reproducibility

- All natural-image conditions are repeated splits of the SAME 1,797 MNIST-like sklearn digits images; no independent datasets or language test. Diabetes role is derived from one true label; forecast role is synthetic. Results are exploratory.
- Source/payload/independent FP64 audit for MLP-ROLE-001 and MLP-HYBRID-001 passed in the CPU container; current MLP-USE-002 likewise verified 96 saved inference payloads and four complete replay trainings. See study READMEs and original archive evidence.
- Fixed updates != equal wall clock/compute; eager CPU rotations are noticeably slower, not intrinsically optimized.
- Do not merge to main, revise past FAIL results, or update canonical MA registry based on these tests.
- **H:** non-linearly factorized reusable roles; **T:** role-separated real datasets + strong native controls + fresh worlds; **D:** only digit attributes had a relative gain vs simple gain, not the strongest head; **C:** linear head suffices; **U:** few splits, single digits corpus, overfitting and CPU overhead.

Full artifact archives in conversation: `MLP_MIRROR_USE_20261017_SOURCE.zip` (SHA256 `3e46efb46f1ae046240f308d679da87575d1b8f2c67dc0b0ee541f354b62030f`), `MLP_MIRROR_USE_20261017_FULL_EVIDENCE.zip` (SHA256 `ecb826c39d6e8eaec6a230eaea0cb1b311933365df0648cb71fbb297cb5df449`). Additional readout-first source and payloads are separately downloadable as `MLP_USE002_SOURCE_20261017.zip` (SHA256 `8ff4d52f3bc535ec6c91f3c9c0eb15b72bf317cc71fe3ab7b14c9700e24d59cb`) and `MLP_USE002_FULL_EVIDENCE_20261017.zip` (SHA256 `c59803da715fba592ed07f9a5a1c1df5b054df60c976d03b6b03457f3c585c26`).

**ERROR CHECK:** Reject extrapolating aligned synthetic gains to natural FFNs, 3 splits to independent populations, 4 product addresses to independent function capacity, 700 updates to converged capacity, 32 bytes of scalars to cheap eager runtime, or validation-chosen checkpoint to test-optimized stopping.