# Research State

## 1. Current formulation

Functional Quotient Compression (FQC) is no longer based on the hypothesis that a reversible “mirror” transform is itself a source of compression. The current formulation is:

> Identify degrees of freedom that are equivalent under the target function/task, quotient or derive those redundancies, represent the remaining shared/private structure with a low-description decoder, and optimize the actual serialized codec under a task-quality constraint.

A useful conceptual pipeline is:

`canonical / functional alignment -> quotient or hard shared structure -> private exceptions -> task-sensitive geometry -> joint codec optimization -> exact serialization -> task witness`

## 2. Established corrections

The following are hard constraints on future claims:

1. **Invertible transforms are not fundamental rate reduction.**
2. **Actual serialization is authoritative.** Headers, padding, metadata, maps and decoder prerequisites count.
3. **Task coupling matters.** Independent per-block distortion can produce false hard-budget conclusions.
4. **Low energy is not low task value.** Spectrally tiny modes can remain decision-critical.
5. **Search work is not codec rate.** Scheduler savings do not count as model compression unless they change the serialized decoder DAG.
6. **Toy optimality is not real-model evidence.**
7. **Weight reconstruction quality is not task quality.** T273-T278 produced a real-checkpoint counterexample in which lower parameter-space error did not imply better KL/NLL.
8. **KL improvement is not sufficient evidence of NLL improvement.** T266-T282 therefore retain both metrics and do not promote KL-only wins to quality claims.

## 3. Main research results

### 3.1 Functional / decision quotient principle

Multiple lines support an observational-equivalence view: gauge freedoms can make different parameterizations functionally equivalent; decoded-signature states with identical future decoder capability can be merged; and synthetic decision-null examples show that some absolute changes can leave all relevant decisions invariant.

This motivates treating the compression target as an equivalence class of task behavior rather than the raw parameter vector alone.

### 3.2 Serializer-aware exact optimization

The exact toy series established that serializer overhead, cross-block terms, decoder prerequisites, tree topology and precision choices must be optimized jointly. These remain deterministic/synthetic mechanism results rather than Transformer compression ratios.

### 3.3 Task-sensitive geometry

Synthetic spectral experiments and the later real-checkpoint affine-baseline experiments agree on a narrow but important conclusion: parameter-space or spectral energy alone is not a sufficient proxy for downstream task value.

### 3.4 Hard 64x certificate framework

For a 16-bit scalar baseline with `N` unique paid scalars, the hard 64x target is `B <= floor(N/4)` bits. Roots, maps, selectors, metadata, coefficients, support descriptions and private residuals are paid unless deterministically derived from already-paid acyclic decoder state.

### 3.5 Real-checkpoint transition: T266-T282

The project has crossed an important engineering boundary since the earlier repository state.

Established on the supplied TinyStories checkpoints:

- an **actual serialized whole-model Transformer artifact** can be produced, independently decoded and run end-to-end;
- all 108 learned tensors of the 28M checkpoint were reconstructed through the codec path used in the experiment lane;
- checkpoint-free decode/forward paths were exercised on representative artifacts;
- serializer/decoder corruption and configuration-mismatch tests were added;
- long-context, causal-mask and KV-cache consistency checks were added;
- deterministic same-environment reproduction was strengthened through artifact hashes and reruns.

The evidence boundary is equally important:

- the tested **64x whole-model candidate failed task quality badly**; therefore 64x quality-preserving compression remains unestablished;
- selective private correction improved the failed 64x candidate, but increasing support beyond the selected level could make quality worse;
- activation-aware affine quantization produced a stronger non-sharing baseline and must now be treated as a control rather than FQC novelty;
- a frozen 28M storage-oriented candidate was smaller than the corresponding non-sharing control and improved KL on the later public-prompt pilot, while NLL evidence remains preliminary and the pilot is not the official TinyStories validation set;
- lossless outer compression changed actual byte ordering between equal-width logical codecs, reinforcing the rule that **final serialized file length is authoritative**.

## 4. Reproducibility state

The historic handoff reconstruction remains imperfect, but the current real-model lane is substantially stronger than the earlier repository snapshot.

For T266-T282 the retained evidence includes:

- source/checkpoint hashes and environment manifests;
- deterministic serializer and independent decoder checks;
- fresh artifact reconstruction checks for the T273-T278 suite;
- 153 regression/integrity tests by T282;
- separate-process reevaluation of frozen candidates;
- long-context and cache-consistency checks;
- compact result/provenance summaries suitable for git, with large checkpoints and large binary artifacts deliberately excluded from canonical history.

This is **same-environment reproducibility**, not independent third-party replication. Official Hugging Face runtime parity, official TinyStories validation evaluation, Apple MPS/CUDA performance and independent-training-seed replication remain open.

## 5. Current evidence boundary

| Claim | Current status |
|---|---|
| Invertible mirror alone creates fundamental compression | Rejected |
| Functional / decoded equivalence is a useful organizing principle | Supported by theory + toy/synthetic evidence |
| Serializer-aware exact optimization can change hard-budget decisions | Exact toy evidence |
| Low parameter/spectral error implies low task error | Rejected; synthetic and real-checkpoint counterevidence |
| Actual serialized Transformer codec can be built and independently decoded | **Demonstrated in the T266-T282 real-checkpoint lane** |
| Selective private correction can help a severe low-rate failure | Supported on the tested 28M candidate; not universal |
| Stronger activation-aware non-sharing quantization improves the internal baseline | Supported on the tested checkpoints; not claimed as FQC novelty |
| FQC functional sharing beats a strong non-sharing baseline at matched final bytes | **Unknown; next central experiment** |
| Official TinyStories validation quality advantage | Unknown |
| Actual serialized Transformer codec beats strong published baselines | Unknown |
| Quality-preserving 64x real-Transformer compression | **Not demonstrated; tested simple 64x family failed** |
| MPS/CUDA runtime or VRAM advantage | Unknown |

## 6. Immediate research transition

The main line should now stop treating implementation existence as the bottleneck. The next gated sequence is:

1. establish the official TinyStories validation + official-runtime evaluation path;
2. freeze a strong activation-aware **non-sharing control**;
3. compare `control -> +functional sharing -> +private exceptions -> +joint allocation` at matched **final serialized bytes**;
4. trace a multi-rate frontier rather than testing only 64x;
5. repeat the direction of effect across 1M/3M/8M/28M before making scale claims;
6. only after a scientific quality win, benchmark MPS/CUDA runtime and memory.

The central falsifiable question is now whether FQC-specific sharing/quotient structure adds reproducible rate-distortion value beyond a strong non-sharing codec.
