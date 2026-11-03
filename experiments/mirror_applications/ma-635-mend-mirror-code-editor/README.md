# MA-635 — MEND editor predicts a compact Mirror code

Status: SCREENING; prior art PA137 MEND. Dedicated branch: `research/ma-635-mend-mirror-code-editor-20261103`.

## H

Given gradient-factor features, an editor that predicts a compact code for a shared rank-2 edit basis will match a generic MEND editor's held-out edit efficacy with less serialized state and compute. A Mirror-specific claim must beat a native low-rank MEND editor with the same code dimension and decoder.

## T

Synthetic 8D gradient features map to 32D rank-one edit outputs on a deterministic nonlinear rank-2 teacher. Train identical 8→32-hidden tanh editors for 1,000 Adam updates: one predicts full 32D updates, one predicts two code values decoded through a shared basis. Native low-rank MEND is the same code model and decoder. Use 2,048 train and 256 audit feature rows; seeds 63501/63502; report actual serialized editor+decoder+base bytes, efficacy nRMSE, locality, active operations and evaluation latency. Fresh 63511–63513 remain sealed on native low-rank alias.

## D

Pending.

## C / U

Counter-hypothesis: a compact MEND output is ordinary low-rank editor distillation; the code model can be exactly represented as a native low-rank MEND editor. Synthetic updates do not establish real factual-edit success.
