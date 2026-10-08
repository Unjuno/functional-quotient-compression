# MA-286 — Cheap-LoRA Mirror column-subspace views

Status: **FAIL for the registered ≥10% Mirror-specific address advantage; aligned sharing mechanism passes its quality/storage screen**  
Branch: `research/ma-286-cheap-lora-mirror-column-20261008`  
Base commit: `28194ea`  
Prior art: PA29, Cheap-LoRA / structured sparse and circulant LoRA

## H — hypothesis

A shared low-rank output factor plus a compact subspace code might encode multiple task maps with less actual payload than task-local Cheap-LoRA, while unrelated maps need private state. A Mirror-specific claim also had to beat a one-hot address over the same shared factor by at least 10% complete-payload bytes.

## T — execution

The synthetic task family has eight distinct aligned rank-4 functions, each selecting a different four-column subspace, and two unrelated rank-4 maps. A shared 16×32 factor is fit from training examples using eight charged task codes. The unrelated tasks receive private rank-4 factors after the fixed validation threshold. Development seeds were 28601/28602; fresh seeds 28611–28613. Each task has 256 train, 128 validation and 512 held-out inputs. Fits use least squares/SVD; optimizer updates: zero.

Controls: hard shared fixed-subspace, fixed and random Cheap-LoRA selectors, matched task-local Cheap-LoRA, one-hot shared-factor gating, and independent full matrices. The fixed Walsh basis is identified in charged architecture metadata; the selector dictionary, per-task addresses, factors, private state, field names, shapes and codec headers are serialized. Quality and throughput are replayed from deserialized payloads.

## D — decision

Across three fresh worlds, integer-code Mirror used 4412 B at mean normalized held-out MSE 7.310e-16, with eight distinct aligned task codes and two private rank-4 fallbacks. Matched task-local Cheap-LoRA used 6580 B at 7.310e-16 MSE; independent full matrices used 41279 B with effectively zero error. Mirror payload was 0.671× matched task-local cLA and 0.107× independent full, passing the registered broad quality/storage screen.

The one-hot shared-B control was functionally equivalent on all tasks at 4490 B and 7.310e-16 MSE. Integer addressing saved 78 B, or 1.74%, below the preregistered 10% Mirror-specific gate. Its measured CPU throughput was 32.4M vs 21.4M examples/s; the compute proxy was 1900544 vs 1900544, equal by construction. The throughput is an eager NumPy microbenchmark, not a deployment kernel. The Mirror address is a compact encoding of the one-hot choice, not a new function family beyond that gate.

**Status: FAIL** for Mirror-specific advantage. The screen still demonstrates that the shared factor plus task subspace addresses saves substantial payload versus task-local factors, while two unrelated maps require private factors.

## Fact / interpretation / hypothesis

**Fact:** All eight aligned maps and two private rank-4 maps had normalized held-out MSE below 1e-14 in all three fresh worlds. Mirror was 4,412 B, one-hot shared-B 4,490 B, matched task-local cLA 6,580 B, and independent full 41,279 B. Five tests pass; all 24 method/world payload hashes and quality metrics replay exactly.

**Interpretation:** Most storage improvement comes from sharing the physical low-rank factor across selected column views. Integer task addresses save only 78 B versus a one-hot gate, so the registered Mirror-specific margin was not reached. Private low-rank factors remain necessary for unrelated tasks.

**Hypothesis:** Address compression may matter more when the selector family or number of logical task codes is much larger, but such a test must compare full payload bytes and routing/compute costs.

## C — strongest counter-hypothesis

One-hot gating computes the same task functions with the same shared factor and essentially the same compute proxy. The integer code's 78 B payload edge is a representation-format win that may vanish under bit-packed or indexed native gates.

## U — unresolved

Task addresses were supplied and charged, not learned from examples. No learned task router, natural Cheap-LoRA workload, optimizer efficiency, continual learning, Transformer, language model, or capacity frontier was tested. The Walsh codebook is a deliberately aligned synthetic basis.
