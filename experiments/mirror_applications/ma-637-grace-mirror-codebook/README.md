# MA-637 — GRACE discrete codebook with shared value basis

Status: SCREENING; prior art PA139 GRACE. Dedicated branch: `research/ma-637-grace-mirror-codebook-20261103`.

## H

A shared rank-4 value basis compresses the values in a growing GRACE-style discrete key/value adaptor while preserving retention and deferral-radius locality. Mirror-specific value requires beating native PCA coding and accounting for codebook growth and lookup/decode time.

## T

Use a synthetic 8D key space and 16D edit values drawn from a shared rank-4 subspace. Fit the basis on 16 calibration values; store 128 sequential edits and evaluate snapshots at 16/32/64/128 entries. Compare explicit GRACE values, Mirror basis+codes, and native PCA basis+codes; all share the same keys, radii, edit IDs, router and base executor. Measure complete actual bytes, bytes/edit, old-edit retention, positive hit rate, negative false positives, lookup+decode latency. Fresh 63711–63713 sealed on native alias.

## D

Pending.

## C / U

Counter-hypothesis: ordinary PCA compression of GRACE values explains all storage gains; key/radius state and retrieval dominate as the bank grows. This synthetic screen does not establish lifelong language-model editing.
