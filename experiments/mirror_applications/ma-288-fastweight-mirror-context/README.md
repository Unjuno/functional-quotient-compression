# MA-288 — context-programmed Mirror views over fast-weight memory

Status: **PROMISING, scoped to aligned memory families**  
Branch: `research/ma-288-fastweight-mirror-context-20261008`  
Evidence lane: MECHANISM / MEMORY / STORAGE / RUNTIME

## H — hypothesis

A dynamic session coordinate can recover four logical fast memories from one physical matrix when the memories differ by a small Givens conjugation. Unrelated sessions need private state. Mirror-specific value requires a better quality/byte/compute frontier than same-byte FiLM and low-rank residual controls.

## Selection

The pool was `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-288, MA-296, MA-299]`. `secrets.randbelow(8)` returned index 5 and selected MA-288. No live MA-288 remote branch existed after fetch.

## T — execution

Each of four sessions receives eight one-hot key/value writes to an 8×8 fast-weight matrix using the delta rule. The shared slow/base memory is learned from session 0. Aligned session matrices are Givens conjugations of that base; the dynamic cue carries one float32 angle. Independent matrices have no relationship to that cue. Each matrix defines a held-out linear memory function on 512 query vectors.

Controls are ordinary shared FWP without a context view, hard tying, one-scalar FiLM, rank-4 per-session residuals, and independent per-session FWP. Development seeds 28801/28802 selected residual rank 4 by all-method held-out output MSE (0.079991 vs 0.085457 for rank 2 and 0.091619 for rank 1). Fresh seeds 28811–28813 used that frozen rank. All context coordinates, session records, tensors, headers and fast states are charged in actual serialized inference payload bytes. Payloads reconstruct the evaluated states. There are no optimizer updates; FWP writes are counted separately.

An initial development screen used an unnecessary base matrix in the independent condition. It is preserved as `DEV_RANK*_INITIAL_BASE_INVALIDATED.csv`; corrected development and all fresh worlds use session 0 as the shared base. Initial fresh tables from before the runtime-path equivalence check are retained as `FRESH_PRE_RUNTIME_PATH_CHECK.csv`.

## D — decision

**PROMISING for the aligned storage/quality frontier; the preregistered per-call runtime gate failed.** Mirror reached near-exact aligned recall with 378 B of serialized state versus 1,203 B for independent FWP, but its direct coordinate path was slower than the rank-4 residual control. Independent contexts exposed the need for private state.

| Method | Aligned MSE | Aligned bytes | Writes | MAC proxy | Direct-path examples/s |
|---|---:|---:|---:|---:|---:|
| Shared FWP, no view | 0.02347 | 317 B | 32 | 4,096 | 153.9M |
| Hard tie to session 0 | 0.02388 | 313 B | 8 | 1,024 | 153.0M |
| Mirror context angle | 1.36e-16 | 378 B | 8 | 1,280 | 22.6M |
| Scalar FiLM | 0.02206 | 383 B | 32 | 1,280 | 114.2M |
| Rank-4 residual | 2.37e-16 | 1,551 B | 32 | 3,072 | 53.2M |
| Independent FWP | 1.10e-16 | 1,203 B | 32 | 4,096 | 157.5M |

Mirror used 0.314× independent FWP bytes and one quarter its FWP writes. It was slightly smaller than FiLM while recovering aligned functions that FiLM could not. Rank-4 residual reached similar quality at 4.1× Mirror's state size and 2.3× its direct-path throughput. On unrelated sessions, Mirror MSE rose to 0.257; rank-4 residual reduced it to 0.0237, while independent FWP reached 1.05e-16.

## Runtime decomposition

The registered eager coordinate path executes Givens input and output views on every query. Its fresh throughput was 0.43× rank-4 residual on average and below the 0.5× gate in all three seeds. A post-audit diagnostic materialized the logical session matrices once from the same serialized state, then served 512 queries/session. With all four cached matrices resident (1,024 B workspace) and 72 μs mean state reconstruction, Mirror reached 201.0M examples/s, versus 172.6M for rank-4 residual and 174.0M for independent FWP. At one query/session it reached 26.5M examples/s. Mean state reconstruction took 65 μs. The cached path improves long-session throughput but its extra 1,024 B resident workspace changes the memory frontier; this diagnostic does not replace the preregistered per-call result.

## Fact / interpretation / hypothesis

**Fact.** Serialized-state reconstruction and query-path outputs replay within 2e-7 maximum absolute difference. The fresh Mirror quality, byte and write-count results are stable in all three worlds. The registered direct-path speed ratio to rank-4 residual ranged from 0.178 to 0.456. The cache diagnostic stores four temporary 8×8 matrices and is reported separately.

**Interpretation.** One angle is enough to multiplex aligned session memories and can cut persistent bytes by 68.6% versus independent FWP. The view is slower when recomputed per query. Session caching can recover throughput after enough queries, at a workspace cost.

**Hypothesis.** A fused kernel or one-active-session cache may keep the persistent storage gain while reducing the runtime and workspace penalty; it requires a separate protocol and fresh seeds.

## C — strongest counter-hypothesis

The session cue is constructed from the same Givens angle used to make the teacher memories, so this is a favorable aligned case. A learned context encoder may not recover such a coordinate, and rank-4 residual storage is larger partly because the shared base and residual factors are separately represented.

## U — unresolved

No language model, long recurrent sequence, learned cue generation, noisy/overlapping writes, or GPU kernel is tested. Cached runtime is exploratory and does not overturn the registered direct-path throughput miss.
