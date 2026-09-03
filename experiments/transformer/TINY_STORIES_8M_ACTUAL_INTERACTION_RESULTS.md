# TinyStories-8M actual interaction results

This note records high-value **actual-model** interaction experiments run locally in the container on the uploaded TinyStories 8M GPT-Neo checkpoint. The checkpoint was evaluated with a manual PyTorch GPT-Neo forward and a locally reconstructed GPT-2 byte-level BPE tokenizer. No GitHub Actions experiment was used.

## Scope

- model: uploaded TinyStories 8M GPT-Neo checkpoint;
- hidden size 256, 8 layers, 16 attention heads;
- fixed four-text probe set;
- candidate codec action: 32-scalar block VQ on WQ/WK with shared layer-family codebooks;
- primary interaction experiments use K=128 (7-bit block index); K=64 is used as a replication point;
- KL is always measured against the unmodified checkpoint on the same next-token probe.

These results are task-distortion evidence for candidate actions. They are not a full emitted codec or a 64x quality certificate.

## 1. Independent action costs are systematically optimistic

For WQ-only K=128 actions, all 28 two-layer combinations were evaluated.

`KL(i,j) / (KL(i) + KL(j))`:

- minimum: `0.9713`;
- median: **`1.1134`**;
- maximum: **`1.2414`**;
- 21/28 pairs were more than 5% super-additive.

K=64 replicated the pattern:

- pair median ratio: **`1.1207`**;
- 22/28 pairs were more than 5% super-additive.

Thus independent per-action task costs systematically understate joint damage in this actual Transformer experiment.

## 2. Pairwise corrections explain low-order joint damage very well

At K=128, all WQ triples and quadruples were exhaustively evaluated.

### Triples

- 56/56 evaluated;
- median joint/additive ratio: **`1.1838`**;
- maximum: `1.3700`;
- 55/56 exceed additive cost by more than 5%.

Using the measured pair interaction excesses,

`D_pairwise(S) = sum_i D(i) + sum_{i<j} [D(i,j)-D(i)-D(j)]`,

triple prediction has:

- median absolute relative error: **`0.67%`**;
- maximum error: `1.60%`;
- predicted-vs-actual ranking Spearman: **`0.99986`**.

K=64 replicated this: median pairwise-corrected triple error `0.67%`, maximum `1.72%`.

### Quadruples

- all 70 evaluated;
- median joint/additive ratio: **`1.2518`**;
- all 70 exceed additive cost by more than 5%;
- pairwise-corrected median relative error: **`1.96%`**;
- maximum: `3.53%`;
- ranking Spearman: **`0.99983`**.

## 3. Higher-order terms are mainly saturating in this action family

Exhaustive WQ combinations were continued through all 8 layers.

| simultaneous WQ actions | median actual / additive | median actual / pairwise | median pairwise relative error |
| ---: | ---: | ---: | ---: |
| 3 | 1.184 | 0.994 | 0.67% |
| 4 | 1.252 | 0.981 | 1.96% |
| 5 | 1.332 | 0.966 | 3.52% |
| 6 | 1.394 | 0.951 | 5.20% |
| 7 | 1.463 | 0.933 | 7.20% |
| 8 | 1.509 | 0.911 | 9.81% |

The dominant pattern is therefore:

- positive pair interactions make independent costs optimistic;
- higher-order interactions become increasingly negative/saturating as more actions are applied.

A simple action-count correction fitted only on 3/4/5-action results reduced pairwise-prediction median error to about `0.64%` at six actions, `0.63%` at seven actions, and `1.56%` at eight actions. The coefficient is dataset-specific and is not a general theorem.

## 4. Same-layer Q+K coupling reverses the interaction sign

A separate K=128 experiment evaluated every Q-layer × K-layer combination.

### Cross-layer Q_i + K_j, i != j

- median joint/additive ratio: **`1.0884`**;
- generally super-additive.

### Same-layer Q_l + K_l

All eight layers were strongly sub-additive:

- ratio range: **`0.6162–0.7230`**;
- median: **`0.6552`**.

K=64 replicated the result in all eight layers:

- range: `0.5824–0.6986`;
- median: **`0.6479`**.

Thus interaction sign depends on the architecture relation between actions. Q and K from the same bilinear attention operator behave as a coupled bundle, unlike Q/K actions in different layers.

## 5. Mechanism: the bilinear cross term cancels part of the score error

At frozen baseline pre-attention activations, the same-layer score perturbation was decomposed exactly as

`dS = dQ K^T + Q dK^T + dQ dK^T`.

Across layers/probes:

- mean cosine between the K first-order term and bilinear cross term: **`-0.949`**;
- mean cosine between the Q first-order term and cross term: `-0.562`;
- including the bilinear cross term reduced score-perturbation norm to an average **`84.1%`** of the norm from the two first-order terms alone.

This gives a direct score-level mechanism for the observed same-layer Q/K functional sub-additivity.

## 6. Optimizer consequence: high global correlation is not enough near the optimum

Sixteen equal-rate K=128 actions were considered: Q0..Q7 and K0..K7. All `C(16,2)=120` two-action choices were evaluated.

- additive individual-cost ranking vs actual joint ranking Spearman: `0.9806`;
- top-5 overlap: only **`1/5`**;
- top-10 overlap: `6/10`;
- top-20 overlap: `18/20`.

Category median joint/additive ratios:

- Q+Q: `1.1191`;
- K+K: `1.0859`;
- cross-layer Q+K: `1.0884`;
- **same-layer Q+K: `0.6552`**.

Five same-layer bundles (`q3+k3`, `q2+k2`, `q1+k1`, `q0+k0`, `q4+k4`) appear in the actual best-10, because an independent additive model systematically overprices them.

The practical QCO implication is narrow but important: **architecture-coupled bundle proposals and pairwise task interactions should be represented explicitly.** Independent block costs are useful for screening, but they are not sufficient for final hard-budget selection.

## Boundaries

Do not infer that:

- all Transformer compression errors are pairwise-dominated;
- the fitted action-count saturation correction transfers to other models or codec families;
- pairwise correction replaces exact replay at commit time;
- Q/K sub-additivity applies to arbitrary Q/K perturbations;
- these experiments establish a real 64x compressed model.

The evidence supports a specific optimizer design pattern: independent costs for cheap screening, explicit architecture-aware bundles and pairwise interactions for candidate selection, then exact replay for commit decisions.
