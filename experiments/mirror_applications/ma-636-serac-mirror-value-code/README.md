# MA-636 — SERAC explicit memory vs shared value-code decoder

Status: SCREENING; prior art PA138 SERAC. Dedicated branch: `research/ma-636-serac-mirror-value-code-20261103`.

## H

A shared rank-4 value basis plus compact edit codes can reduce SERAC's explicit counterfactual memory while preserving scope routing, edit efficacy and locality. Mirror-specific value requires beating ordinary PCA/low-rank value coding and must include retrieval/decode latency.

## T

Synthetic 8D query space, 128 edit keys, 16D counterfactual values with a true shared rank-4 subspace; 96 edits train the basis and 32 are held out. Each key gets in-scope positive and out-of-scope negative queries; a nearest-key router with frozen radius is shared by all methods. Compare explicit SERAC values, shared-basis Mirror codes, and native PCA value codes. Measure actual full payload bytes, edit nRMSE, scope hit/false-positive rates, and retrieval+decode latency. Fresh 63611–63613 sealed on native alias.

## D

Pending.

## C / U

Counter-hypothesis: this is ordinary low-rank compression of SERAC counterfactual values. Synthetic scope queries do not establish real editing behavior.
