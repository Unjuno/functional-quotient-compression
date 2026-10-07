# Mirror-MoE remains an open research lane

Date: 2026-10-07 JST  
Status: active architecture hypothesis, not yet experimentally established.

## Why this is separate from the current residual-View result

MS008–MS012 established a strong current sensor rule:

- calibrate observations toward a canonical latent;
- keep raw sensor identity out of the shared world core;
- use a small residual View only when calibration leaves measurable mismatch;
- Backprop trains the shared core and View codes.

That architecture is conditional and parallel across sensors, but it is **not yet a sparse Mixture-of-Experts** if every addressed View that is required for the batch is evaluated.

Mirror-MoE is therefore preserved as a separate hypothesis.

## Proposed Mirror-MoE

The world model remains shared.  The MoE lives in a residual functional layer, not as separate world models.

For canonical latent h:

    h_shared = F_theta(h)
    h_out = h_shared + alpha * R_{m(h)}(h; theta, c_m)

where:

- theta is shared by every expert;
- c_m is a low-description Mirror code;
- R_m is a structured residual View generated from the same shared parameters;
- m(h) selects one or a small top-k subset of Mirror experts;
- total stored expert cost should scale mainly with codes/bases, not K copies of FFN weights.

A valid Mirror-MoE claim requires active compute to stay approximately bounded with K through sparse selection.  Evaluating all K Views and averaging them is a multi-view ensemble, not sparse MoE.

## Routing rule

Sensor identity should not be the primary router input. MS008 showed that exposing sensor identity directly to the shared core can create sensor-conditioned pathways and damage cross-sensor law transfer.

Preferred order:

1. sensor-specific encoder/calibration -> canonical latent;
2. shared world computation;
3. router from canonical latent / world state / task-relevant role;
4. top-1 or top-2 residual Mirror experts.

Initial experiments should use deterministic/oracle role addresses to isolate expert representation. Learned routing should be added only after the expert mechanism survives matched controls.

## What must be compared

At matched actual serialized bytes and controlled active compute:

1. shared-only world core;
2. shared core + fixed gate residual;
3. shared core + FiLM / low-rank residual;
4. dense parallel Mirror residual bank;
5. sparse top-k Mirror-MoE;
6. independent-expert MoE upper control.

Primary measurements:

- local prediction quality;
- cross-sensor source-only law transfer;
- worst-view quality;
- active forward/backward compute;
- actual serialized bytes;
- router entropy / expert occupancy;
- expert ablation and wrong-expert degradation.

## Main hypothesis

A sparse Mirror-MoE can create multiple functional experts from one shared parameter set, obtaining specialization without duplicating the world model.

This is distinct from claiming that Mirror creates Shannon information or that K Mirrors equal K independent experts.

## Failure conditions

Reject the Mirror-MoE lane if any of the following persist:

- sparse routing gives no gain over a fixed gate or low-rank residual at matched bytes/compute;
- expert specialization improves local prediction but degrades cross-sensor law transfer;
- useful quality requires evaluating most experts, removing the sparse-compute advantage;
- expert codes/bases grow until storage approaches an ordinary MoE;
- routing depends mainly on sensor identity instead of canonical world state.

## Current recommendation

Do not merge Mirror-MoE into the validated sensor architecture yet. Keep the validated architecture as the baseline and test sparse Mirror-MoE as an orthogonal extension.

The most promising version is:

    calibration
      -> canonical latent
      -> shared world core
      -> state/role-conditioned top-k residual Mirror bank
      -> shared prediction

This preserves the possibility that Mirror is useful not only as a sensor residual coordinate, but also as a low-description shared-parameter expert family.
