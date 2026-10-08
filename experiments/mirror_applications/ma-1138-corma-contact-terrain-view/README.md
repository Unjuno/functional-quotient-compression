# MA-1138 — CoRMA contact × terrain View

Status: **NOT ESTABLISHED before protocol freeze**
Branch: `research/ma-1138-corma-contact-terrain-view-20261008`
Baseline: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`
Draw 12: selected uniformly from 556 eligible P0/UNTESTED rows; complete pool and replay details are in `source/selection_pool.csv` and `source/random_draw.json`.

## H — Hypothesis

A small Mirror coordinate over a shared contact-rich adaptation policy could represent useful contact-regime × terrain functions with lower policy/state cost than native CoRMA/RMA adaptation, while preserving slip and locomotion quality. This candidate was selected, but the hypothesis was not experimentally tested.

## T — Feasibility screen only

Read PA402 (CoRMA) and PA401 (RMA), the MA-1138 blueprint and robot-dynamics research notes. The execution container has no robot and none of the tested simulation/control stacks was installed. CoRMA/RMA policy code, pretrained weights, and their environment assets are not in this repository. PyTorch `2.6.0+cpu` reports CUDA available: `False`. Dependency availability is captured in `VERIFICATION.json`. No dataset, policy, simulator, or task metrics were accessed.

## D — NOT ESTABLISHED

The required CoRMA/RMA native control and contact-rich rollout environment cannot be reproduced here. A generic hand-written toy simulator would not decide the registered CoRMA insertion hypothesis or satisfy its strongest control. This is an execution blocker, not evidence for or against Mirror. The MA remains **UNTESTED** in the registry.

## C — Strongest counter-hypothesis

The Mirror coordinate may add no useful freedom beyond CoRMA's existing contrastive context, and any apparent gain could come from conditioning capacity or privileged contact information.

## U — Unknown

All quality, bytes, latency, adaptation compute, contact-regime retention, and held-out terrain behavior are unmeasured. No protocol or model hyperparameters were frozen because the mandatory control/environment were unavailable.

## Provenance

Random selection seed: `b51d370a31f67d6c00e4885ee6a029e171e1dac5e7dfa327445192b7038e22c3`; zero-based index: 542; pool SHA-256: `1cc85874a9224be8dc5dd36b816c1b6ee8bf8b3c54ab1a5a1c65d2161b7139c1`. No audit/fresh data were opened.
