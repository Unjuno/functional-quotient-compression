# MA-383 — L2P prompt pool with Mirror prompt generation

Status: **FAIL; development gate missed, fresh seeds remain unopened.**
Evidence lane: QUALITY / STORAGE / COMPUTE / RETRIEVAL
Base commit: `0739bf3`
Prior art: PA56, L2P prompt pool retrieval.

## H — hypothesis

For task prompts on a shared Givens orbit, one physical prompt plus a per-slot angle preserves retrieval-conditioned continual classification with fewer bytes than explicit prompts and exceeds a same-sized scalar basis. Unrelated prompts should require private state.

## T — protocol and execution

Protocol was frozen before development (`PROTOCOL.json`). Two development seeds (38301, 38302), 8 slots, 32-dimensional inputs, 4 classes, 1,600 Adam updates and 204,800 sampled examples per method. Same nearest-centroid Euclidean retrieval is used by explicit, scalar, Mirror, and hypernetwork controls; task IDs are hidden at inference. Retrieval keys and frozen classifier are charged in each deterministic ZIP/NPY FP16 inference payload. Each seed also ran the unrelated-prompt boundary condition. Fresh seeds 38311–38313 were not opened because the registered development gate failed.

## D — decision

**Fact:** Retrieval accuracy was 1.0 for every method and condition. In aligned seed 38301 test accuracy was explicit 0.8008, scalar 0.9109, Mirror 0.8396, hypernetwork 0.7959. In aligned seed 38302 it was 0.7632, 0.6960, 0.7151, 0.7432, respectively. Actual payload sizes were 2,172 B explicit, 1,970 B scalar, 1,970 B Mirror, and 5,477 B hypernetwork. Thus Mirror used 90.7% of explicit bytes, missing the registered <=80% threshold; it lost to scalar by 7.1 pp in one seed and beat it by 1.9 pp in the other. Mirror training wall time was 3.46s and 3.19s versus scalar 1.04s and 1.16s. Per-query prompt-generation proxy was 256 MACs for scalar and Mirror, 18,432 for the hypernetwork; nearest-key retrieval was 256 distance multiply-accumulate terms/query for all methods. Unrelated results are retained in the result JSON and CSV.

**Interpretation:** The preregistered aligned quality/storage gate fails. The run does not support a Mirror-specific advantage: at equal serialized bytes, scalar performance varied by seed and Mirror was slower to train. Shared orbit structure alone did not assure that the learned coordinate recovered useful prompt behavior under this frozen classifier.

**Hypothesis:** A prompt view may help when task prompts are known to occupy a stable low-dimensional orbit and coordinate optimization is robust; this synthetic run did not establish that condition as sufficient.

## C — strongest counter-hypothesis

The toy teacher and fixed classifier create an optimization-sensitive classification problem; the independent explicit prompt pool is not consistently best either. The negative result establishes failure of this registered protocol and gate, not failure of prompt views generally.

## U — unresolved

No natural L2P benchmark, continual stream, forgetting metric, fresh confirmation, convergence-capacity result, or optimized inference kernel was tested. No capacity claim is made.

## Verification

Run `python -m pytest -q experiments/mirror_applications/ma-383-l2p-mirror-prompt-pool/tests`. Result payloads are in `results/development/`; the implementation writes each inference archive with prompts/codes, keys, classifier, metadata, and archive headers included in its measured size.
