# Global Dense Mirror architecture

Status: **active Phase II architecture hypothesis**

## Core idea

Keep a single dense model as the primary computation path. A single context-aware router controls a small bank of global Mirror states that change how that dense model is used.

The target is not to turn a few blocks into conventional MoE experts. The target is:

```text
context
  -> one router
  -> global Mirror state / mixture
  -> one dense model
```

The dense path remains wide. Conditional capacity is introduced through a low-description control state rather than duplicated expert matrices.

## Three independent resources

These must not be conflated:

1. **Dense width** — shared feature capacity available to every input.
2. **Mirror breadth** — how many independent functional directions one state can modify.
3. **Mirror count** — how many alternative states the router can choose or mix.

MN010 showed that increasing Mirror count with an 8-dimensional control was ineffective on the tested task, while a 256-dimensional direct global modulation produced a measurable signal in one data world.

## Global Mirror state

For the current two-layer prototype, one rich state directly modulates:

- 32 attention residual channels per layer;
- 64 FFN hidden channels per layer;
- 32 FFN output channels per layer.

This gives 256 scalar degrees of freedom per state.

Future breadth-controlled experiments should use a low-description fixed basis so that a breadth of 32/64/128/256 changes the state payload without adding a large learned decoder matrix.

A suitable prototype basis is a fixed orthogonal Hadamard-derived basis over the 256 modulation coordinates. This is a research control, not a claim that Hadamard is task-optimal.

## Router

The router must consume a causal context summary that preserves task-relevant order.

The current prototype uses a causal exponentially decayed prefix summary. The critical requirement is functional: the router input must distinguish contexts that require different Mirror states.

The router is a single model component. Increasing the number of states increases its output width, but does not create a sequential router chain.

## Function-preserving state split

For soft routing, split one state (j) as:

```math
a_j \rightarrow (a_j-\log 2,\ a_j-\log 2)
```

```math
m_j \rightarrow (m_j+\varepsilon v,\ m_j-\varepsilon v)
```

At insertion time, the mixed control is unchanged.

The split direction (v) should be chosen from a development-only task-gradient covariance or a decoder-known structured basis. Audit data must not influence the direction.

## Growth policy

State growth is allowed only through a paired comparison from the same parent checkpoint:

- candidate A: add one Mirror state;
- candidate B: no structural change, same extra optimization budget;
- candidate C: dense widening with similar added serialized bytes.

Use the same data/batch schedule where possible.

A state is adopted only when development evidence shows additional value above the no-change shadow and the byte-matched dense-widening alternative.

## Stopping rule

A valid stopping rule is an optimization rule, not a capacity theorem.

The next experiment will use multiple disjoint development minibanks instead of one short fine-tune estimate. State growth stops after two consecutive proposed additions fail the preregistered development criterion.

## Evidence boundary

The architecture is motivated by MN009-MN010. It remains a small synthetic prototype. No claim is made that this is already better than production Dense or sparse-MoE architectures.
