# MA-257 — Compositional Mirror context group

Status: **PROMISING in a synthetic compositional task family**. This is not broad capacity evidence.

## H

A shared 32-dimensional classifier with two Mirror factor angles should recover an unseen (1,1) task from the other three, match a non-Mirror additive task-angle control, and use fewer actual serialized bytes.

## T

Protocol was frozen before development (`PROTOCOL.json`, base `c0a232b`). Each world contains four binary tasks created by applying two pairwise orthogonal rotations to one random linear teacher. Training excludes the whole (1,1) combination from both compositional models. Controls: tied shared classifier; PA16 fixed-sign PSP bound from three independently fitted observed task models; independent all-task reference; additive per-task angle table with held-out angle `theta10+theta01-theta00`; factorized Mirror with two learned factor coordinates. Two development worlds (25701/25702) and three fresh worlds (25711–25713) were evaluated without changing the frozen setup.

## D

**PROMISING.** Factorized Mirror held-out (1,1) accuracy was 0.9937 and 0.9900 on development and 0.9861, 0.9858 and 0.9875 on fresh worlds. Additive control accuracy was 0.9880, 0.9900, 0.9858, 0.9824 and 0.9895. Every world met the frozen >=0.90 accuracy and <=0.03 gap gates. Mirror serialized payload was 819 B versus 827 B for the additive control in all five worlds. PSP scored 0.764 and 0.715 mean accuracy on the three observed tasks and does not support the withheld task. The independent four-task oracle reached about 0.981 mean test accuracy; it uses held-out training labels and is an upper reference.

## C

The strongest counter-hypothesis is that the result comes from the task generator's exact additive rotation group, not a general property of Mirror contexts. The non-Mirror additive-angle control reached nearly identical task accuracy. The actual storage improvement over that control is only 8 bytes in these small archives.

## U

This does not establish composition on unrelated task families or neural networks. No broad capacity claim follows from four possible factor combinations. The useful evidence is held-out function quality on a deliberately compositional synthetic family and a small actual-byte saving over a matched code table.

## Fact / interpretation / hypothesis

- **Fact:** All five worlds meet gates; all 25 payloads reload and replay, with maximum FP16 accuracy difference 0.000163; three tests pass.
- **Interpretation:** Two factor coordinates recover a genuinely withheld task in this generated family and store 8 fewer bytes than the 3-angle additive control. Quality is indistinguishable at this scale.
- **Hypothesis:** Factorized group coordinates may be useful when task transforms truly compose; natural task transfer remains untested.
