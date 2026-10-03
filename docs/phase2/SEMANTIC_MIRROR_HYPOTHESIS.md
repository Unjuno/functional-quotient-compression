# Semantic Mirror hypothesis

Status: **auxiliary Phase II research lane; open hypothesis, not an established result**

## Current role

MN007-MN008 tested whether related concepts can reuse shared parameter identity plus a low-description state/view coordinate.

MN008 showed that explicit functional recombination supervision can strongly improve held-out recombination and semantic grouping. However, a shared-base additive control also improved and remained stronger on median NLL in both tested worlds.

Therefore semantic Mirror is **not** the current main architecture claim. It remains a useful side lane for studying reusable representations and supervision.

The active Phase II architecture is [Global Dense Mirror](GLOBAL_DENSE_MIRROR_ARCHITECTURE.md).

## Original hypothesis

Conventional embeddings represent semantic similarity largely through geometric proximity. The stronger structural question is whether some related concepts can instead be represented as:

```math
\text{concept} \approx (\text{shared parameter identity},\; \text{small view/state coordinate})
```

or, in the dual direction:

```math
\text{concept} \approx (\text{concept coordinate},\; \text{shared transformation basis})
```

## What MN007-MN008 established

- shared-base/state representations can be trained and serialized;
- explicit recombination supervision can make factors reusable across held-out combinations;
- semantic grouping and functional recombination can improve together under additional relation supervision;
- shared transformation directions can also align with state-like factors.

## What remains unestablished

- task loss alone does not reliably recover the intended semantic quotient;
- relation supervision contains additional semantic information;
- phase Mirror did not beat strong additive factorization in MN008;
- natural-language semantic compression is untested.

This lane should only become central again if a Mirror parameterization improves a matched rate-quality-composition frontier beyond additive and low-rank controls.
