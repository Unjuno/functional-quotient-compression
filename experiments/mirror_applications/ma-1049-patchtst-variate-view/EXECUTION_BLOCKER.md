# MA-1049 execution blocker — source semantics

## H — frozen hypothesis
A two-scalar per-sensor rotation-scale View over a shared channel-independent PatchTST would improve late-window sensor MSE by at least 2% against hard sharing and beat byte-matched diagonal FiLM.

## T — attempted
- Created dedicated branch `research/ma-1049-patchtst-variate-view-20261115` from worker-ready MA-1048 integration commit `430f93c1a`.
- Read PA329 and froze `PROTOCOL.json` before downloading/opening data.
- Source: `jimmygao3218/PEMS08`, pinned revision `7d5f30e2f1b538973ff805fc84ac42a0bd7422aa`.
- `PEMS08.csv`: 295×3 edge list (`from,to,cost`), SHA256 `e5ab2a62f275741e6b07d3fba1623883ff9d692d406a44724d26efddc5e24b20`.
- `PEMS08.npz`: 18,525,419 bytes, member `data`, shape `(17856,170,3)`, float64, SHA256 `e1d03ce74e9fb79149e6e7c37c680214822c4a285baa2f42278cacabcd25c075`.
- README contains no column definitions beyond license/citation. The source does not document which of its three channels means speed.
- No training, development score, or audit score was computed. No fallback feature was selected after inspection.

## D — NOT ESTABLISHED
This source cannot validly test the speed-forecasting hypothesis without guessing a channel after seeing the schema. This is a source-semantics blocker, not evidence that PatchTST Mirror fails.

## C — strongest counter-hypothesis
One of the three unnamed channels may be traffic speed, and the dataset may have an established conventional order documented externally. That convention was not provided with the frozen source, so selecting it now would be an unregistered post-inspection choice.

## U — unresolved
A future amendment/new MA could pin an authoritative PEMS08 channel schema (or a different labeled traffic-speed source) before downloading data. Then rerun the frozen comparison with genuine independent sensor targets.
