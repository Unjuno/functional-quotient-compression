# Evidence Policy

This repository separates claims by evidence type. A stronger lane may cite a weaker lane as motivation, but a weaker lane must not be presented as stronger evidence.

## Evidence lanes

| Code | Lane | What it can establish |
|---|---|---|
| `THEORY` | Mathematical / structural result | A theorem, bound, identity, or conditional derivation under stated assumptions |
| `EXACT_TOY` | Exhaustive or deterministic toy experiment | Correctness/mechanism for the tested finite system |
| `SYNTHETIC` | Simulated scheduler / routing / geometry experiment | Empirical behavior inside the declared synthetic environment |
| `SERIALIZED_CODEC` | Actual encoded payload | True byte/bit cost for the implemented codec on the tested object |
| `REAL_MODEL` | Real neural-network experiment | Model/task behavior on the declared checkpoint and evaluation protocol |
| `RETRACTED` | Invalidated / superseded result | Historical information only; cannot support an active claim |

## Claim rules

### 1. Rate claims

A hard compression claim must use the serialized payload, including all required paid decoder state. Logical or estimated bits may be used only for search/proposal.

For a 16-bit scalar baseline and `N` unique paid scalar values, the hard 64× budget is:

`B_total <= floor(N / 4)` bits.

`B_total` must include every paid root, map, selector, codebook, support description, coefficient, metadata field, alignment/padding cost, and private residual needed by the decoder.

### 2. Derived state

A quantity may be charged at zero payload bits only when it is deterministically derived from already-paid or explicitly external-fixed state through an acyclic decoder protocol. Convenience, internal generation, or formula availability is not sufficient.

### 3. Task-quality claims

A real-model candidate must be evaluated under a predeclared quality protocol using the same checkpoint, dataset/task definition, and baseline contract. A toy or synthetic quality metric cannot substitute for a real-model witness.

### 4. Search-compute claims

Search work, query cost, validation cost, calibration work, or scheduler work are reported in their own units. They are not codec bits.

### 5. Optimality language

- `global optimum` requires exhaustive search, a valid certificate/gap, or a proof over the stated feasible family;
- `best tested` means only the best among tested candidates;
- a finite candidate-family optimum is not a global learned-representation optimum;
- a local surrogate ranking is not a proof of hard-budget improvement.

### 6. Spectral / magnitude language

Low magnitude, low Fisher, low singular energy, low pairwise commutator, or a common eigenbasis is not by itself evidence that a component is safely removable. Downstream task/decision sensitivity must be checked.

### 7. Reconstructed artifacts

Artifacts recreated from specifications/logs rather than recovered byte-for-byte from the original run must be labeled `RECONSTRUCTED`. Reproducing the recorded result upgrades confidence, but does not make the reconstruction the original artifact.

## Hard do-not-resurrect claims

The following claims are explicitly disallowed unless new evidence overturns the current state:

- invertible Mirror/basis changes alone create fundamental rate-distortion gain;
- common eigenbasis implies common small support;
- formula-derived bases are automatically 0-bit;
- arbitrary synthetic scheduler savings update codec rate bounds;
- D70–D120 synthetic work is real Transformer compression evidence;
- near-rank-2 spectral energy implies rank-2 decision sufficiency;
- the empirical fixed shrink `q=0.75` is globally optimal or pathwise safe;
- the existing research establishes 64× compression on a real Transformer.

## Required metadata for new experiments

Every new experiment should record at minimum:

- experiment ID;
- hypothesis;
- evidence lane;
- parent claim(s);
- git commit;
- model/data/config hashes where applicable;
- seed(s);
- environment/dependency information;
- serializer version for rate claims;
- raw outputs;
- decision: `PASS`, `FAIL`, or `UNCERTAIN`;
- claim-ledger changes caused by the result.
