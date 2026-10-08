# MA-443 — support-set generated Mirror code

Status: SCREENING  
Evidence lane: FEW-SHOT / ADAPTATION / STORAGE  
Base commit: `1914ab0`

## H — falsifiable hypothesis

A support-set encoder that initializes a low-dimensional Mirror code, followed by a small number of code-space updates, will reach held-out task quality at least as well as gradient-only Mirror initialization and full-vector adaptation while lowering per-task adaptation steps and total amortized bytes.

> **Mirror insertion:** this experiment adds a support-set-generated task code `m` to a shared backbone so task specialization can be amortized into a small encoder and code instead of a private parameter vector.

## PA75 / PA76 delta

PA75 supplies gradient adaptation from a shared initialization; PA76 supplies a learned task latent and decoder. MA-443 combines an encoder-produced Mirror initialization with gradient refinement and charges the encoder amortization, decoder/base, code, and inner-loop state. It is distinct from MA-442's zero-initialized Mirror code test.

## Protocol

Use the same 8D Givens-orbit regression family as MA-442 for paired comparison. Compare zero-init Mirror gradient adaptation, support-encoder initialized Mirror adaptation, encoder-only code, full-vector MAML, and rank-two LEO-style latent decoder. Development worlds 44300/44301 choose encoder/code LR and refinement steps; fresh worlds 44310/44311/44312 with 20 held-out tasks each. Report query NRMSE at 0/1/3/5 refinement steps, encoder support-set FLOPs, actual serialized payload bytes per task and amortized per-task bytes at N=1/20/100 tasks.

PASS: encoder-initialized Mirror matches full MAML within 1.10x NRMSE and beats zero-init Mirror in at least 2/3 fresh worlds, with lower total amortized bytes at N=20 and no higher query compute. FAIL if the encoder does not improve over zero-init or loses after encoder bytes are amortized.

## C — strongest counter-hypothesis

A generic latent encoder or one direct gradient step may give the same initialization; charging the support encoder may erase any per-task savings.

## U — unresolved

Natural task sets, nonlinear neural backbones, and support-set distribution shifts remain untested.

## Results and decision

**D — FAIL under the registered storage/compute gate.** At five refinement steps, fresh mean NRMSE: encoder-init Mirror 0.2746, zero-init Mirror 0.3889, full-vector adaptation 0.2759, and LEO-style latent 0.1452. Encoder-init Mirror beat zero-init in all three worlds and matched full adaptation within 1.10x. However, its actual task payload was 2,566B and shared encoder checkpoint 3,520B; at N=20 amortized total was 2,742B versus full adaptation 2,348B and LEO 2,602B. Query/adaptation wall was 0.001388s vs 0.000420s for full adaptation (about 3.3x).

**FACT:** the support encoder gave a large improvement over zero-init Mirror and reached near-full-MAML quality. It did not lower amortized bytes or query cost against the stronger controls. Encoder-only NRMSE was 0.3848; refinement was needed for most of the gain. LEO was more accurate but also used more bytes than full adaptation.

**INTERPRETATION:** a learned initializer is useful for placing `m` near the task orbit, but its persistent encoder and refinement compute erase the intended compression/adaptation advantage in this screen. This does not support a Mirror-specific compression claim.

**H:** tested whether amortized support-set code initialization plus refinement improves over zero-init Mirror and matches full adaptation at lower amortized cost.

**T:** same 8D Givens-orbit task family as MA-442; 8 support/128 query samples; 0/1/3/5 refinements; 300 first-order meta updates × 8 tasks; dev selected outer LR 0.03 Mirror encoder and LEO, 0.01 other methods; fresh 3 worlds × 3 seeds × 20 tasks. Encoder bytes were serialized once per world/seed/method and amortized at N=1/20/100.

**C:** tasks are deliberately aligned to Givens rotations; the poor full-vector and zero-init baselines may depend strongly on support sampling and first-order meta-training.

**U:** natural tasks, nonlinear backbones, encoder distribution shift, and optimized batched encoder/adaptation kernels remain untested.
