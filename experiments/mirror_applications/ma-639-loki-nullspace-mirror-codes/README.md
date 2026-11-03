# MA-639 — LOKI null-space projection, then Mirror value coding

Status: SCREENING; prior art PA140 LOKI. Dedicated branch: `research/ma-639-loki-nullspace-mirror-codes-20261103`.

## H

Applying LOKI's protected-key null-space projection before compressing edit values separates interference reduction from storage compression. A shared Mirror basis should compress projected updates while preserving edit efficacy and protected-key locality, and must beat ordinary PCA on the projected values.

## T

Synthetic 32×32 layer, 16 rank-one edits, 16 protected keys, and four held-out edits. Compare unprojected full ROME updates, LOKI-projected full updates, and projected updates encoded at ranks 0/1/2/4/8/16 for aligned rank-2 and private rank-16 value worlds. Fit bases on 12 edits only. Measure update efficacy, protected-key leakage, complete payload bytes and operation proxy. Fresh 63911–63913 sealed on native PCA alias.

## D

Pending.

## C / U

Counter-hypothesis: the null-space projection provides locality while ordinary PCA provides compression; Mirror adds no distinct effect. Synthetic keys do not establish real LOKI editing.
