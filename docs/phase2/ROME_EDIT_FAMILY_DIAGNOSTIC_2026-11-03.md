# ROME shared-basis edit family diagnostic — 2026-11-03

## Trigger and scope

MA-631 and MA-632 are consecutive ROME edit-bank screens that both failed for the same reason: shared Mirror coordinates were ordinary low-rank update representations and the stronger native low-rank/additive controls matched or dominated them. Pause the remaining directly adjacent ROME/MEMIT codebook candidates MA-633 and MA-634 until their protocol names a degree of freedom that differs from standard low-rank update compression.

This is not a conclusion about model editing generally. Other editor families with different mechanisms remain in the queue.

## Evidence

- **MA-631:** In a synthetic 32D layer with 16 rank-one edits, efficacy error was <2e-14 and one-hot-key locality leakage 0. Mirror edit state was 1,358–1,360 B versus 2,587–2,616 B independent, but complete payload was 5,390–5,399 B versus 6,621–6,648 B (about 81%), missing the frozen <=50% gate. Plain shared low-rank was smaller (5,277–5,285 B), and native additive factorization exactly aliased Mirror. Fresh sealed.
- **MA-632:** Shared/private ranks mapped the quality/bytes boundary. For true private ranks 0/2/4, Mirror recovered exact efficacy with ranks 0/2/4 but missed the frozen <=70% full-payload gate. At rank 8, the payload exceeded independent updates. At true rank 16, 12 training edits did not identify the private span; residual nRMSE remained 0.276–0.537 while independent edits were exact. Native joint low-rank had the same reconstruction quality/locality and 400–460 fewer complete bytes than the split Mirror representation. Fresh sealed.

## Interpretation

The initial shared-basis compression is real against independent rank-one values, but splitting a joint low-rank code into Mirror shared/private coordinates adds metadata/serialization overhead and no new function. Private dimensions restore quality only while their basis and per-edit coefficients remain cheaper than storing direct updates; the frontier crossed by true rank 8 in this screen.

## Resume condition

Resume MA-633/634 only with an explicit structural distinction from joint low-rank update coding, and use the same parent edit set, serializer, locality metrics and actual-byte accounting. Pretrained-model ROME/MEMIT efficacy and generalization are still untested.
