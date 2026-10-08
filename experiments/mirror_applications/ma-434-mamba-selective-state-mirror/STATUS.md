# MA-434 status: FAIL (quality improves; storage/runtime gates fail)

## H — falsifiable hypothesis

A shared input-selective SSM plus one role angle would reproduce four role dynamics at <=60% of independent SSM bytes, similar quality, and the same recurrent steps.

## T — executed

Four roles, sequence length 16, 2D state/input. The teacher uses input-dependent Δ=softplus(wΔx+b), shared stable decay modes conjugated by role rotations, and shared B/C maps. Compared shared SSM, Mirror role angles, role-specific Δ scalar gates, and independent per-role decays/B/C. 300 AdamW updates × batch 32; 2 development worlds × 3 seeds selected LR 0.01; 3 fresh worlds × 3 seeds. Fresh sequences used the same per-world teacher dynamics with disjoint sequence samples. Actual serialized state bytes were measured.

## D — FAIL

Fresh mean sequence NRMSE / bytes / training wall: shared 0.13162 / 2,585B / 0.794s; Mirror 0.02235 / 2,837B / 9.026s; delta gate 0.12879 / 2,837B / 0.832s; independent 0.03894 / 2,649B / 0.873s. Mirror substantially improves sequence quality over independent, but actual bytes are 7.1% higher rather than <=60%, and measured training wall is about 10.3× the independent control in this unfused implementation. All methods execute 16 recurrent steps. The registered compression gate fails.

## C — strongest counter-hypothesis

The small 2D state and many tiny tensors make serialized container/key metadata larger than the raw parameter savings. Mirror's per-sample matrix conjugation is also implemented as unoptimized tensor operations, inflating measured wall time. This runtime is implementation-specific and not a fused-kernel estimate.

## U — unresolved

Higher-dimensional SSMs, fused selective scans, real token sequences, perplexity, inference throughput, and whether code/state savings amortize at production scale remain untested. Synthetic Mamba-like proxy only.

