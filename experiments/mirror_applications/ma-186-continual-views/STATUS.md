# MA-186 status

- Status: **FAIL** for the registered aligned quality gate; storage and retention subclaims passed.
- Branch: `research/ma-186-continual-views-20261007`
- Base commit: `480d956ef9eb84216358a72d809cc81bcd16a958`
- Result commit: `e2a54862a926f024254b1e29cc2101c5bdf826b9`
- Verification commit: pending
- Development complete: yes
- Fresh/audit opened: yes, after selecting LR 0.01 on development seeds
- Results committed: yes
- Verification committed: yes (included in tracker commit)
- Registry row updated: yes

## H — hypothesis

A frozen common backbone plus one learned task-specific Givens angle will preserve aligned skills at <=1.10x rank-2 LoRA final task MSE and <=0.25x its incremental inference bytes per skill. Unrelated task maps should require private parameters.

## T — execution

Five-task 16D->8D synthetic linear stream; 600 Task 0 pretraining updates and 300 per subsequent task. Development seeds 18601/18602 compared LR .003/.01; .01 selected by the registered all-method/all-condition mean final MSE. Fresh seeds 18611–18613 used .01. Controls: sequential full-weight fine-tune, rank-2 LoRA, shared generated rank-2 hypernetwork, hard tie, and independent full maps. Inference uses deterministic packed records; resume uses complete `torch.save` optimizer checkpoints. A1 changed the inference serializer before fresh access; A2 corrected matched-format baseline accounting before fresh access.

## D — decision

**FAIL**: fresh aligned Mirror/LoRA final-MSE ratios were 1.85, 1.23, 1.05; threshold <=1.10 was missed in 2/3 worlds. Mirror passed the byte threshold in 3/3: 20 B/skill versus 241 B/skill, with 707 B total versus 1,590 B for LoRA. Earlier-task MSE increases were 0 in 3/3. Mirror throughput was about 0.35x LoRA. Unrelated tasks failed under the one-angle view; independent full weights were much more accurate and larger.

## Strongest counter-hypothesis

The fresh quality miss could be due to the restrictive one-angle coordinate or optimizer conditioning rather than a general limitation of task-specific Mirror views. The absolute errors are near 1e-11, so the relative 1.10 threshold magnifies small differences; the preregistered rule still governs.

## U — unresolved

The hypernetwork control was degenerate: both low-rank factors initialize at zero, yielding zero gradients for both factors and matching hard tying. Therefore this run cannot establish superiority over a competent hypernetwork. Runtime used eager PyTorch on a small CPU harness. No natural continual-learning or language-model evidence; no capacity claim.

## Deviations and decisions

- Initial generic-serializer development and first A1 mismatched baseline results are preserved in `DEV_TORCHSAVE_DIAGNOSTIC.csv` and rerun before fresh access.
- A2 matched serializer baseline accounting was frozen before fresh access; both development rates were rerun.
- No post-fresh tuning was done. Further variants require a new MA/amendment and fresh seeds.
