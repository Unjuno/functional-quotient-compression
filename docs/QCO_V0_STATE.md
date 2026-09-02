# Quotient Codec Optimizer v0 — Integrated State

This integration layer combines previously separate exact mechanisms into one small-instance state space.

A candidate consists of:

- a global/local layout or tree description;
- one codec option per block;
- private payload bits;
- conditional shared PAID prerequisites such as roots/dictionaries/bases;
- precision/mode metadata;
- an exact joint task-error callback.

Rate is computed as:

`layout/private payload + block private payload + paid bits(union of decoder prerequisite closure)`.

The task objective is intentionally **not** forced to be a sum of per-block errors. This preserves the E2 lesson that cross-block coupling can invalidate a separable distortion model.

## Why this integration is needed

E7 demonstrates a coordinate dead zone: layout alone and precision alone can fail to improve an incumbent while their joint move improves it. D63 shows that private/shared choices are coupled by one-time root costs. E2 shows that task distortion may itself be coupled.

The canonical exact toy solver therefore enumerates the whole small joint state and uses:

- exact serializer/decoder-DAG accounting for rate;
- exact joint callback for task error;
- hard serialized-bit feasibility before objective comparison.

This is not yet the scalable optimizer. It is the **oracle/reference solver** against which future branch-and-bound, DP, pricing, and surrogate methods should be tested.

## Next extension

The next scalable layer should expose admissible lower bounds for:

1. remaining private bits;
2. not-yet-opened shared prerequisite closures;
3. task error under partial assignments;
4. root-family pricing/replacement bounds from D64–D69.

Only then should branch-and-bound replace exhaustive enumeration on larger real-model pilots.
