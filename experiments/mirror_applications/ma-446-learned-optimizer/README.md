# MA-446 — Learned optimizer for Mirror coordinates only

## H — Hypothesis

A learned per-step diagonal update schedule acting only on a 2D Mirror coordinate will reach at least 10% lower fresh query NRMSE than development-tuned Adam and SGD after four updates.

## T — Planned test

Synthetic 2D linear regression tasks with four support examples and 128 query examples. Compare zero code, tuned SGD, tuned Adam, and a learned coordinatewise step schedule. Development IDs 100–199 and worlds 44600–44601 select all settings; fresh IDs 2000–2059 and worlds 44610–44612 remain sealed until freeze. Actual serialized policy and task-code bytes are measured.

## D — Pending

Protocol frozen; no numerical result yet.

## C — Strongest counter-hypothesis

For a convex two-dimensional least-squares objective, ordinary tuned SGD/Adam already provide a strong update rule; a short learned schedule may only repackage a learning-rate choice.

## U — Unknown

Whether learned coordinatewise update rates improve held-out adaptation quality, and whether that advantage survives policy-byte accounting.
