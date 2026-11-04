# MA-640 — RRDA edit/suppress dual roles on shared adapter basis

Status: SCREENING; prior art PA141 RRDA. Branch: `research/ma-640-rrda-dual-role-mirror-views-20261104`.

## H

A shared adapter basis with role-specific Mirror coordinates can represent RRDA's edit and suppress functions using less edit-state storage when role updates align. As angle between role updates grows, private basis rank should be needed. Mirror-specific value requires beating native shared-basis gating.

## T

Synthetic 16D adapter outputs with two RRDA roles; vary role-vector angle 0/15/30/60/90 degrees. Compare independent full adapters, shared rank-1/rank-2 Mirror role codes, and native rank-1 gate / rank-2 shared-basis controls. Same router/role labels and base executor for every method. Measure edit and suppress nRMSE, full payload bytes, role-state bytes and operation proxy. Fresh 64011–64013 sealed on native alias.

## D

Pending.

## C / U

Counter-hypothesis: role-specific scalars on a shared LoRA basis are standard gating, and two-dimensional cases reduce to ordinary low-rank factorization. Synthetic roles do not establish RRDA performance on a language model.
