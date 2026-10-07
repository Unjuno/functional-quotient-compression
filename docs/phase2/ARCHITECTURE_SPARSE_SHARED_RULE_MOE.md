# Sparse Compositional Shared-Rule Transformer

Date: 2026-10-07
Status: **current architecture hypothesis**

## 1. Design principle

Do not Mirrorize the whole Transformer by default.

Partition functionality into:

1. **always-shared computation** — common structure that can be stored once;
2. **shared sparse rules** — reusable specialist residuals selected per token/context;
3. **truly private residuals** — only if evidence shows they cannot be represented economically through the shared bank.

The target is not "choose one smaller expert." The target is "compose several reusable rules into a virtual expert."

## 2. Block form

For hidden state (h),

[
F(h)=F_{shared}(h)+sum_{jin S(h)} a_j(h)R_j(h),
]

where:
- (F_{shared}) is the canonical shared FFN path;
- (R_j) is a reusable rule atom;
- (S(h)) is a sparse active set;
- (|S(h)|=k) may be greater than one.

The same rule atom may be reused across domains, tasks, sensors, or contexts.

## 3. Preferred Transformer placement

```text
x
|
LayerNorm
|
Shared Attention
|
residual
|
LayerNorm
|
Shared FFN Base -------------------+
|                                  |
|                           sparse rule router
|                                  |
|                         choose multiple atoms
|                                  |
+------------------------- add specialist residual
|
next block
```

Initial experiments specialize the FFN residual path, not attention and not the entire block.

## 4. Rule atom families

Keep these conceptually separate:

### A. Low-rank residual atom

[
R_j(h)=U_jV_j^T h
]

This is the non-Mirror baseline and must remain in every Mirror-specific comparison.

### B. Mirror residual atom

A shared nonlinear FFN is perturbed by low-description structured coordinates such as stretch and shear around the activation.

Mirror coordinates must be paid for in serialized bytes unless deterministically derived from already-paid state.

### C. Full independent expert

This is the upper-flexibility control. It is not expected to be storage efficient, but it is necessary to show whether quality can be recovered by unrestricted private capacity.

## 5. Routing hierarchy

Routing research proceeds in stages:

1. oracle rule addresses;
2. supervised address learning;
3. LM-loss-only sparse routing;
4. routing from ordinary contextual hidden states without explicit rule-token scaffolding.

Do not mix routing failure with representation failure.

## 6. Sparsity

Top-1 is not the default assumption.

Current synthetic evidence strongly favors composing all active rule factors. Search (k) independently from the number of stored atoms (M).

Key diagnostics:
- quality;
- exact retained-rule count;
- address consistency;
- cross-domain reuse;
- atom redundancy;
- actual active compute.

Load balance is diagnostic, not a success criterion.

## 7. Training

Current default:
- Backpropagation;
- no ES for continuous shared/rule parameters;
- no recurrent Mirror loop;
- overcomplete rule bank may be trained first and pruned later;
- near-convergence capacity and fixed-compute learning efficiency are reported separately.

## 8. Storage contract

Count:
- all learned tensors;
- rule codes / coordinates;
- routers;
- indices and codebooks;
- non-derived structured bases;
- required metadata.

Use actual serialized payload length. Parameter count is only descriptive.

## 9. Success hierarchy

### Architecture success
Shared-rule composition beats strong standard MoE / Dense controls at a useful byte-compute-quality frontier.

### Composition success
Held-out or ordered rule combinations work without storing one independent expert for each combination.

### Mirror-specific success
Mirror rule atoms beat low-rank/shared-rule controls at comparable bytes and compute.

### LLM success
The above survives natural-language training with no synthetic rule labels and with realistic throughput.

None of these higher claims follows automatically from the earlier one.
