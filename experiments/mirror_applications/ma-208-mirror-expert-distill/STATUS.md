# MA-208 status

- Status: SCREENING
- Branch: `research/ma-208-mirror-expert-distill-20261007`
- Base commit: `621dec2ffdf55c3f0503dd75695ea27fdc68544c`
- Protocol frozen: yes (implementation and protocol committed before development; fresh seeds remain sealed)
- Development complete: no
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## H — hypothesis

Specialist teachers in one hidden Givens orbit can be distilled into a shared student and task angles with lower actual bytes than ordinary per-task student heads, while independent teachers require private weights.

## T — setup

Synthetic five-task four-class MLP. Compare hard tie, sequential KD, private output heads, rank-2 LoRA, Mirror hidden views, independent full students and teacher upper references. Development seeds 20801/20802; fresh 20811–20813 remain sealed.

## D — pending

Targeted MoE-to-dense and multi-teacher distillation prior art was read. Protocol and implementation are in progress.

## C — strongest counter-hypothesis

Ordinary per-task output heads can exactly express the same aligned hidden rotations and may be a simpler control with acceptable total bytes.

## U — unresolved

No results. The teacher family is synthetic and deliberately related to the candidate view.
