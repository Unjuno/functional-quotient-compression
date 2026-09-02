# Model-wide K128 block-VQ rate skeleton

This note records deterministic rate arithmetic using the certified SmolLM2-135M scalar inventory. It is **not a quality result** and does not establish 64× feasibility.

## Scope

Certified paid scalar counts used here:

- attention: `26,542,080`
- MLP: `79,626,240`
- attention + MLP: `106,168,320`
- other / norm scalars: `35,136`
- hard 64× target: `4,203,594` bytes

The candidate nonembedding primitive uses 32-scalar blocks and a power-of-two shared codebook.

## K256 is serializer-fragile

At `K=256`, each 32-scalar block requires exactly 8 index bits (`0.25 bit/scalar`).

With the relatively light conservative `PQ32/K16` embedding layout and the lightest tested one-codebook / no-alignment nonembedding configuration, total serialized arithmetic is approximately:

- total: `4,199,400` bytes
- headroom to the 64× target: **`4,194` bytes**

Thus K256 leaves essentially no room for additional selectors, richer decoder metadata, unexpected headers, or alignment state.

## K128 is the current robust target

At `K=128`, each 32-scalar block requires 7 packed bits (`0.21875 bit/scalar`).

A deliberately heavier candidate using:

- conservative `PQ16/K256`, 3-bit embedding codebook;
- 40 paid embedding Householder reflections;
- seven nonembedding codebook families;
- 4-bit nonembedding codebook entries;
- per-codeword scale/zero state under the tested serializer convention;
- 4-bit per-tensor scale metadata;
- eight six-bit Householder reflections per nonembedding tensor;
- FP16 norm / other payload;

has deterministic arithmetic:

- nonembedding + norm payload: `3,040,744` bytes
- embedding payload: `883,584` bytes
- total: **`3,924,328` bytes**
- headroom: **`279,266` bytes**

This is a rate skeleton only; it does not imply that K128 reconstruction quality is acceptable on the real model.

## Seven-bit packing is mandatory

There are `3,317,760` 32-scalar nonembedding blocks.

- dense 7-bit packing: `2,903,040` index bytes;
- worst-case extra byte padding per 210 tensors changes this by at most about `210` bytes;
- storing each K128 index as `uint8` uses `3,317,760` bytes.

Under the candidate above:

- dense packing leaves `279,266` bytes headroom;
- `uint8` indices make the total `4,339,048` bytes, **exceeding the 64× target by `135,454` bytes**.

Therefore the current K128 design requires real bit packing. Logical 7-bit accounting is not sufficient unless the serializer actually emits packed 7-bit indices.

## Entropy coding priority

In the current synthetic task-aware K128 VQ experiment, global codeword entropy was approximately `6.990 bits/block`, versus the fixed 7-bit code. The ideal model-wide gain was only about `4.3 KiB` before entropy-model overhead.

For now, deterministic fixed 7-bit packing is the preferred baseline. More complex entropy coding should be reconsidered only if actual real-model codeword usage is materially more skewed.

## Boundary

This document establishes only a feasible **byte-allocation skeleton** under declared serializer assumptions. Real-model task quality, codebook compatibility, support costs, and exact final serialization still require direct validation.