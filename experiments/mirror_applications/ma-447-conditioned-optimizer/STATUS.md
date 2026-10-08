# MA-447 status

**FAIL — conditioning misses quality controls and costs more serialized bytes.**

## H / T

Tested one shared learned 2D-coordinate optimizer conditioned on a domain Mirror code against tuned Adam, one unconditioned learned schedule, and independent per-domain schedules. A1 corrected bytes to include the full conditioned shared base/modulation basis.

## D — Fact

At step 4, mean NRMSE across fresh worlds/seeds: Adam 0.5008, unconditioned 0.7708, conditioned 0.6020, separate 0.6370. Conditioned loses to Adam overall and to separate schedules in domain 0 (0.4098 vs 0.3661). Exact whole-bank N=20 bytes/task: conditioned 151.65B, separate 139.05B, Adam 113.85B.

## C — Strongest counter-hypothesis

Tuned Adam is robust on this task family; a domain-conditioned learned optimizer adds state without beating the native control.

## U — Unknown

Natural task families, nonlinear models, recurrent optimizer policies, and runtime/energy remain untested.
