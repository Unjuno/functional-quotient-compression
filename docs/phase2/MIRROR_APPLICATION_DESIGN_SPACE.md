# Mirror Application Design Space

Date: 2026-10-07 JST
Status: active idea map and experiment backlog

## Core abstraction

The research object is no longer a single "Mirror architecture".

For a physical learned object with parameters theta, introduce a low-description address m:

    F(x; theta) -> F(x; theta, m)

The compression question is whether one physical object plus small addresses can replace several independently stored logical objects at acceptable quality and compute.

    one physical object + small Mirror coordinates
                        |
                        v
              many logical functions

This does NOT imply independent information or independent-model capacity. Every claim must be measured at actual serialized bytes and useful quality.

## What can be multiplied logically?

The registry treats any repeated object as a candidate:
- MoE experts;
- LoRA/adapters;
- attention heads;
- KV/GQA heads and cache views;
- Transformer depth / tied blocks;
- FFN nonlinear roles;
- embeddings / output heads / positional roles;
- multi-token / packet candidates;
- memory banks and retrievers;
- quantization codebooks / centroids;
- holographic binding roles;
- continual-learning skills / update geometries;
- ensembles / students;
- SSM states and execution kernels;
- structured orthogonal / low-displacement matrix families;
- task-delta and model-merging capability directions;
- Bayesian posterior / ensemble subspaces;
- neural operators over PDE/function families;
- relation-specific graph message transforms;
- diffusion control branches/adapters;
- neural-cellular-automata local update rules and goal states.

## Four mechanism classes

### 1. Multiplicity replacement
Replace N physical copies with one shared physical object and N low-description addresses.

### 2. Coordinate multiplexing
Read the same object in different hidden / attention / activation / memory coordinates.

### 3. Functional composition
Activate or compose several addresses at once. Top-1 is not assumed. Sum, signed mixture, product, and sequential composition are separate hypotheses.

### 4. Temporal multiplexing
Use addresses over token phase, packet slot, or depth/time position. TM001 is the first packet-level example.

## Factorized addresses

Do not require one giant address table. Test factorized coordinates such as

    m = (m_expert, m_head, m_depth, m_time)

and measure whether combinations provide useful functional multiplicity without paying for an independent parameter object for every Cartesian-product state.

## Evidence contract

For every candidate report separately:
1. useful task quality;
2. actual serialized bytes;
3. logical multiplicity and physical multiplicity;
4. active MAC/FLOP proxy;
5. training compute / tokens / updates;
6. wall-clock calibration;
7. interference / retention;
8. routing/address overhead;
9. pruning or compaction behavior;
10. strongest simple control.

Never convert the number of possible addresses or combinations into a capacity claim by itself.

## Experiment progression

1. Cheap mechanism screen on controlled tasks.
2. Byte-matched and compute-aware adversarial controls.
3. Fresh-world replication.
4. Integration into the common nanoGPT testbed when relevant.
5. Natural-language tiny-LM validation only after the mechanism survives synthetic controls.
6. Combination tournament only among individually viable mechanisms.

## Current high-value families

The first queue intentionally spans different physical multiplicities:
- Mirror-MoE / shared experts;
- Mirror-LoRA/adapters;
- logical attention heads;
- logical KV/GQA heads;
- virtual depth / tied blocks;
- Mirror-RoPE;
- packet/parallel decoding;
- Mirror quantization;
- holographic/FFT binding;
- continual learning and expert distillation.

The canonical machine-readable backlog is IDEA_REGISTRY.csv under experiments/mirror_applications/.
