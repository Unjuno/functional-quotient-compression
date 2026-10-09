# MA-451 — PathNet path plus Mirror module role

## H — Hypothesis

A small task role coordinate on selected shared PathNet modules improves held-out path quality and reduces bytes versus independent per-task modules.

## T — Test

Synthetic two-layer 4D modular regression, four physical modules per layer. Development selected path-default angles; fresh worlds 45110–12, three seeds, 64 task IDs per seed, four held-out path identities. Mirror infers role angle from support. Actual payloads include physical modules, path IDs, angles, weights and metadata. A2 reports N=1/20/64, matching the fresh bank size; A3 records operation counts without changing quality.

## D — FAIL under registered N=20 gate

At N=20, Mirror recovered held-out role functions (NRMSE 0.000) vs PathNet-only 0.186, but used 158.25B/task vs 146.05B independent modules. At N=64, Mirror used 63.45B/task vs 89.64B, with exact recovery. Mirror query wall was 0.393ms vs PathNet 0.105ms on this CPU proxy.

## C — Strongest counter-hypothesis

The teacher is aligned to Givens roles; independent modules are more storage-efficient at N=20 and PathNet avoids role-search compute.

## U — Unknown

No learned routing, nonlinear modules or language-model evidence.
