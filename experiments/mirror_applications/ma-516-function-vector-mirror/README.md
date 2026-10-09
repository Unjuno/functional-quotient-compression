# MA-516 — Function-vector Mirror compression basis

Status: **FAIL** for useful held-out function preservation under this fixed extraction/injection protocol.

## H / T / D / C / U

**H:** Layer-6 activation deltas extracted from four support examples can be stored as compact task vectors and PCA-compressed while preserving held-out task accuracy at lower intervention bytes than explicit vectors.

**T:** Pinned `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, 124,439,808 parameters, CPU float32. Sixteen arbitrary object-to-color tasks per world; four support pairs and four held-out queries. Function vectors are layer-6 residual-stream deltas between support and neutral prompts. PCA fit on development worlds 51600/51601 × seeds 0-2. Fresh worlds 51610-51612 × seeds 0-2. Controls: query-only, direct ICL, explicit vectors and PCA ranks 4/8/16. Every vector/basis/coordinate and reconstruction object is included in actual intervention bytes. The selected GPT-2 checkpoint plus tokenizer/config files are 550,959,861B.

**D — FAIL.** Direct ICL accuracy by fresh world was 16.7%, 18.8%, and 13.5%, each above 12.5% chance but weak. Query-only accuracy was 0% in every world. Explicit function-vector addition was also 0%; PCA ranks 4/8/16 were 0% in every world. Rank-4 PCA compressed intervention payload from 50,793B to 17,569B (34.6% of explicit) at vector NRMSE .00946, but did not preserve useful task accuracy. Explicit-vector mean target NLL was 6.872 versus 6.762 query-only; direct ICL NLL was 2.533.

**Compute:** Mean per-bank inference time: query-only 1.132s, direct ICL 5.671s, explicit-vector application 1.752s, PCA rank-4 application 1.647s. Extracting the 16 task vectors took 3.040s per bank on CPU; PCA projection was below 0.2ms. Applying vectors was faster than direct ICL but failed the quality test. The checkpoint plus intervention payload is the total deployment footprint; intervention bytes alone are reported for storage comparisons.

**C:** These support-delta vectors did not function as reusable task representations when added at the query position. This is evidence against this extraction/injection method on GPT-2, not against all causal function-vector methods.

**U:** Larger pretrained models, causal optimization of task vectors, natural ICL tasks, other layers/scales, and cross-task vector application interference remain untested.

## Measurement amendment

The first A1 run recorded absolute `perf_counter()` for query-only latency. It is preserved in `artifacts/a1_pre_latency_fix_exploratory.*` and excluded for runtime. Timing was corrected and replayed on the same frozen inputs without changing model/task/ranks; predictions and payloads were unchanged.
