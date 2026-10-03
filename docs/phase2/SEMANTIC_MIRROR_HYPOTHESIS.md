# Semantic Mirror hypothesis

Status: **open hypothesis, not an established result**

## Motivation

Conventional embeddings represent semantic similarity largely through geometric proximity: related concepts tend to occupy nearby points in a learned vector space.

The current Mirror-native hypothesis asks whether some of that redundancy can instead be represented structurally:

[
\text{concept} \approx (\text{shared parameter identity},\; \text{small view/state coordinate})
]

In words: related concepts may be able to reuse the **same underlying parameter object** while differing by a small learned Mirror/view coordinate.

This is stronger than saying two vectors are close. It says the model may be able to factor a concept into a reusable canonical representation plus a low-description transformation.

## Two dual parameterizations

Both directions must be tested.

### A. Shared canonical parameter + concept/state Mirror

[
z_c = M_{m_c}(B_{g(c)})
]

- (B_{g(c)}): shared canonical base;
- (m_c): small Mirror/view code;
- (g(c)): learned or known assignment to a shared base.

This is the direct "same parameter, different view" formulation.

### B. Shared Mirror direction/basis + concept-specific coordinate

[
z_c = M_{A u_c}(b_c)
]

or more generally

[
M_c = \sum_k u_{c,k} M_k
]

where the transformation directions (M_k) are shared and concepts carry only small coefficients.

This is the dual view: the reusable object is the **transformation basis**, not necessarily the canonical concept base.

Neither parameterization should be assumed superior a priori.

## What MN007 established and did not establish

MN007 established that shared-base/state representations can be trained, serialized, decoded, and used for functional tasks. It also showed that they can be smaller than independent embeddings.

MN007 did **not** show that:

- semantic families reliably self-organize into the same base;
- one state code transfers cleanly across semantic families;
- Mirror-Phase beats additive or low-rank factorization;
- forcing a semantically plausible grouping improves task NLL.

In fact, the tested low-rank and additive controls were strong, and semantic-family ARI remained low.

## Core unresolved problem

Task loss alone may have many equally predictive factorizations. A model can lower NLL without choosing the decomposition humans call "semantic."

Therefore the next objective should not merely cluster concepts. It should test **transformation consistency**:

> When two concepts differ by the same semantic relation, does the same low-description state/view operation produce the corresponding functional change across multiple shared bases?

This can be measured directly by state-swap and cross-family transfer tests.

## Proposed decision test

For concept families (f), states (s), and held-out pairs ((f,s)):

1. train on a subset of family x state combinations;
2. require state (s) to induce a consistent functional delta across several families;
3. hold out some family x state pairs;
4. compare:
   - independent embeddings;
   - additive factorization;
   - CP/low-rank factorization;
   - shared-base Mirror;
   - shared-transform-basis Mirror;
   - optional small private residual;
5. match actual serialized bytes;
6. score task NLL, accuracy, swap consistency, cross-family transfer, semantic clustering, and private-residual rate.

The hypothesis is supported only if a Mirror parameterization improves the rate-quality-composition frontier beyond the additive/low-rank controls.

## Relation to the rest of the project

Phase I showed that functional value is not captured by raw parameter distance alone and that shared/private structure must be judged under task-sensitive geometry and exact rate accounting.

Phase II now asks a stronger native-model question: can training *choose* a parameterization in which task-relevant similarities are represented by actual shared parameter identity and low-description state coordinates?

This document records that question. It does not answer it.
