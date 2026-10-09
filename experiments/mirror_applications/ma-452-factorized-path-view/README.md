# MA-452 — Factorized path × Mirror address

## H — Hypothesis

A factorized path ID plus support-inferred Mirror role generalizes to held-out path-role combinations and beats both flat tables and routing controls on storage/quality.

## T — Test

Two-layer 4D regression with four paths × six roles. Eight combinations held out. Fresh worlds 45210–12, three seeds, 64 tasks/seed. Path IDs are provided; Mirror infers role from support. Compare PathNet, Routing Networks-style role index, factorized Mirror, and full per-task matrices. Actual N=1/20/64 payloads include modules and addresses.

## D — FAIL for Mirror-specific claim

At N=20, Mirror and Routing both achieved effectively zero NRMSE on held-out combinations, while PathNet-only scored 0.197. Mirror used 145.45B/task; Routing used 139.05B. Routing matches the function with the same physical modules and role search, but the integer role index is smaller than a float Mirror angle.

## C — Strongest counter-hypothesis

This is ordinary discrete role selection over reusable modules; Mirror adds no unique behavior beyond Routing Networks.

## U — Unknown

No learned path router, continuous role family, or natural-data evidence.
