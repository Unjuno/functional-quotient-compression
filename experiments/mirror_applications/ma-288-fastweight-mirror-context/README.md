# MA-288 — context-programmed Mirror views over fast-weight memory

Status: SCREENING  
Evidence lane: MECHANISM / MEMORY / STORAGE / RUNTIME  
Base commit: `e259f27`

## H — hypothesis

If session memories differ only by a small Givens conjugation, one physical fast-weight memory plus one dynamic session coordinate can recover each logical memory with less state than independent Fast Weight Programmer (FWP) matrices. Unrelated session memories should need private state. Mirror-specific value requires beating same-state scalar FiLM and low-rank residual controls.

## Selection

The random pool was `[MA-274, MA-276, MA-278, MA-282, MA-286, MA-288, MA-296, MA-299]`. `secrets.randbelow(8)` returned 5 and selected MA-288. A live branch check found no remote MA-288 branch.

## T — protocol

Four session contexts each receive eight one-hot key/value writes to an 8×8 fast-weight matrix using the delta rule. The shared slow/base memory is learned from the first context. In the aligned condition, later session memories are conjugate Givens views of that base memory; the continuous session cue supplies one angle `m`. In the independent condition, each session has an unrelated matrix and the same cue carries no useful relationship. Held-out inputs query each session's logical function.

Controls: ordinary shared FWP without a view, hard tying to the base memory, one-scalar context FiLM, per-session rank-2 residuals over the base memory, and independent per-session FWP matrices. Every fast state, dynamic coordinate, context record, tensor and serialization header is paid. The write count, active MAC proxy, wall time and coordinate-path throughput are reported.

Development seeds 28801/28802 selected residual rank 4 (mean all-method held-out output MSE 0.079991 vs 0.085457 at rank 2 and 0.091619 at rank 1). An initial independent-world base initialization mismatch is preserved in `DEV_RANK*_INITIAL_BASE_INVALIDATED.csv`; corrected dev worlds match the preregistered first-context base. Fresh seeds 28811–28813 use the locked rank 4 selection.

## Gates

PASS for the aligned mechanism if all three fresh worlds have Mirror output MSE <=1e-5, total inference state <=0.5× independent FWP, and coordinate-path throughput >=0.5× selected low-rank residual control. Mirror-specific value additionally requires a better quality/byte/compute frontier than the same-scalar FiLM and matched residual control. Independent memories locate the private-state boundary. This synthetic screen is not a Transformer or natural-memory result.

## D — decision

Pending locked development and fresh runs.

## Fact / interpretation / hypothesis

**Fact:** pending.  
**Interpretation:** limited to synthetic fast-weight recall.  
**Hypothesis:** a dynamic context angle can retrieve an aligned family of logical memories from one physical fast state; arbitrary memories cannot.

## C — strongest counter-hypothesis

The view cue is deliberately aligned with how the memory matrices are generated. A simple input/output gate or low-rank update may recover the same state at comparable cost, while conjugation adds inference operations.

## U — unresolved

No language model, long recurrent sequence, learned context encoder, realistic nonstationary write stream, or GPU kernel is tested. The FWP write states use an exact basis-key support episode; longer/overlapping streams may change overwrite and stability behavior.
