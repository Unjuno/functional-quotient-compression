# MA-1051 — iTransformer channel-conditioned Mirror residual

Status: SCREENING. Prior art: PA331.

## Hypothesis
Two per-sensor rotation-scale coordinates on shared variate tokens improve future-window forecasts over cross-variable attention with hard sharing and beat byte-matched FiLM, while preserving robustness to input sensor ablation.

## Insertion
The coordinate is applied to the first two dimensions of each variate token after temporal embedding and before cross-variable attention. The model continues to attend across available sensor tokens. Controls include shared attention, diagonal FiLM, and an independent per-sensor reference. Protocol is in `PROTOCOL.json`.
