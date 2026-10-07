# Worker Research Context — Mirror application program

Date: 2026-10-07 JST
Purpose: compact scientific context for workers after the first verified MA experiments.

## Read this before selecting a new experiment

The project is no longer asking whether "Mirror works" in the abstract.

The current question is:

> What part of real model variation lies on a cheap shared functional orbit, what part needs private residual capacity, and can the orbit be exploited for storage, active compute, cache reuse, or communication?

## Verified evidence

### Works on aligned synthetic variation
- MA-241 tied experts + layer Views: PROMISING.
- MA-244 shared K/V role/head Views: PROMISING.
- MA-245 MLKV + layer Views: PROMISING.
- MA-249 one future head + offset Views: PROMISING.
- MA-250 shared linear expert + role Views: PROMISING.
- MA-691 lazy canonical KV read: exact mechanism PASS / PROMISING.

These were deliberately aligned to the tested View family. They show mechanism feasibility, not prevalence in natural models.

### Fails / boundaries
- MA-253: narrow Mirror failed independent rank-2 private expert variation; cache-safe final-FFN placement itself passed.
- MA-247: recurrent depth Mirror failed the fixed-budget development screen even on an aligned teacher; optimization/credit assignment remains a plausible cause.
- MA-248: Mirror packet code did not beat a simpler broadcast shared code.

## Most important inference

Do not use only all-aligned teachers.

Whenever possible define

    Delta(alpha) = alpha * Delta_view + (1-alpha) * Delta_private

and sweep alpha = 0, .25, .5, .75, 1.

Measure:
- quality;
- actual serialized bytes;
- required private residual rank/size;
- active compute;
- wall clock.

The desired product is an **orbit/private Pareto frontier**.

## Baseline ladder

Use the cheapest meaningful mechanism first:

1. hard tying / no adaptation;
2. scalar or diagonal / IA3 / FiLM;
3. rank-one / BatchEnsemble;
4. shared low-rank basis / VeRA;
5. structured View;
6. structured View + private residual;
7. independent object upper control.

For MoE compression also include MoLAE/MoBE-style shared latent/basis controls when relevant.

## Symmetry/gauge distinction

Some Views change parameters but not the function:
- neuron permutation;
- sign/scale symmetries;
- exact orthogonal computational gauges;
- QuaRot/SpinQuant-style full-precision rotations.

These have **zero new functional multiplicity**.

They can still be valuable for:
- quantization;
- cache layout;
- merging/alignment;
- numerical conditioning.

Always label gauge benefit separately from functional specialization.

## KV-cache result that changes the program

MA-691 verified:

    K_m = K A_m
    V_m = V B_m

can be read exactly from one canonical cache by transforming only the current query and attention output.

The follow-up research shows this extends algebraically to affine and rectangular maps.

Even stronger, if

    q_m = q C_m
    K_m = K C_m^{-T}

then the QK score matrix is exactly shared. With value/output Views, several logical attention functions may share:
- physical cache;
- QK dot product;
- softmax;
- canonical value aggregation.

See:
- `MIRROR_KV_CACHE_REUSE.md`
- `MIRROR_KV_MOE_GENERALIZATION_2026-10-07.md`
- `experiments/mirror_applications/KV_MOE_FOLLOWUP_QUEUE.md`

## MoE compute hypothesis

If

    E_m(x) = W2 psi_m(W1 x)

then a top-k mixture can be evaluated exactly as

    h = W1 x
    z = sum_m a_m psi_m(h)
    y = W2 z

when W1/W2 are genuinely shared.

This pays W1 and W2 once, rather than k independent expert GEMMs.

This is stronger than parameter compression if it survives real kernels.

## Distributed MoE hypothesis

Standard expert parallelism sends activations to GPUs that own experts and returns outputs.

If the expert base is replicated/resident and only tiny logical View codes differ, logical routing may be local. Remote communication may be necessary only for private residuals.

Test the continuum:
- fully aligned -> local View only;
- partially private -> residual-only communication;
- fully private -> standard expert parallelism.

Do not claim communication savings without measured or faithful simulated network traffic.

## Runtime discipline

Structured Views have repeatedly shown eager CPU overhead.

For runtime-sensitive candidates report:
- analytical active compute;
- eager timing;
- compiled/vectorized timing;
- GPU/fused timing when hardware exists;
- materialization/cache bandwidth.

A byte win with a runtime loss remains a storage-only result.

## Storage discipline

Actual serialized inference payload is authoritative.

For caches, also report live-state bytes separately because model-file serializer overhead can hide the real serving effect.

For serving, distinguish:
- one physical allocation aliased by many logical Views;
- several equal tensors copied in memory.

Only the first is zero-copy sharing.

## Current high-value research sequence

1. finish MA-692..700;
2. score-sharing / affine exact cache extensions;
3. orbit/private cache corrections;
4. low-bit canonical cache;
5. physical block aliasing;
6. canonical writer/specialist reader;
7. single-GEMM top-k Mirror FFN;
8. distributed residual-only expert routing;
9. natural-language and GPU gates;
10. only then unified MA-750.

## Worker assistance rule

If a candidate stalls:
- first determine whether the failure is representation, optimization, routing, runtime, or evidence accounting;
- compare the next-cheapest baseline rung;
- use an aligned/private alpha sweep when the family question is unclear;
- do not silently lengthen training or open fresh data to rescue a failed preregistered gate;
- create a new amendment/MA when testing a different hypothesis.

Dedicated experiment branches remain the measurement source of truth.
